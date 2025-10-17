# ============================================================
#  GPT-2 Distributed Training & Inference
# ============================================================
#  Compatible with: PyTorch ≥ 2.0
#  Features:
#     - Supports single-GPU or multi-GPU via torchrun (DDP)
#     - Gradient accumulation for large effective batch
#     - Fused AdamW optimizer
#     - FlashAttention via scaled_dot_product_attention
# ============================================================

import os, math, time
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import inspect

from hellaswag import render_example, iterate_examples


# ============================================================
# 1. Transformer Core Modules
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):

        super().__init__()
        assert config.n_embd % config.n_head == 0

        # dimensions
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_proj.NANOGPT_SCALE_INIT = 1 # normalization

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        ) # lower-triangular mask

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # word token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # word position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # blocks
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # NOTE: the last layer normalization
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        # initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # init linear layers
        if isinstance(module, nn.Linear):
            # gaussian w∼N(0,0.022) (from gpt-2 and BERT)
            std = 0.02
            # tag sacling
            # std=0.02×2×nlayer−0.5​
            # from DeepNorm / GPT-2 scaling law
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= 2 * (self.config.n_layer ** -0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # initialize bias: zero initialize
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # init word and position embeddings
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # B (batch size)
        # T (sequence length)
        # sanity check
        B, T = idx.size()
        assert T <= self.config.block_size

        # add word embeddings and pos embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        # send x to blocks
        for block in self.transformer.h:
            x = block(x)
        # last layer: LayerNorm
        x = self.transformer.ln_f(x)

        # cal loss
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    # -------------------
    # 1.Group the model parameters
    # 2.Determine which ones should have weight decay applied
    # 3.Select the most appropriate version of AdamW
    # (including CUDA-accelerated variants)
    # 4.Finally return the optimizer object
    # -------------------
    def configure_optimizers(self, weight_decay, learning_rate, device,
                             master_process=True):
        # collect params
        # requires_grad is true
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # group params
        # dim >= 2: usually params matrices
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # dim < 2: shape transfer or bias or hypers
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}")

        # fused adamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        if master_process:
            print(f"using fused AdamW: {use_fused}")

        # instace optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate,
            betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# ============================================================
# 2. Data Loader (tokenized batching)
# ============================================================

# fineweb load shards
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, \
        num_processes

        assert split in {'train', 'val'}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards


        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        self.current_position = self.B * self.T * self.process_rank
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])



    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    # -----------------------------------------------------------------------------
    # helper function for HellaSwag eval
    # takes tokens, mask, and logits, returns the index of
    # the completion with the lowest loss

    def get_most_likely_row(tokens, mask, logits):
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm

# ============================================================
# 3. DDP Initialization
# (check if the environment support ddp, else return to initial state)
# ============================================================

# -1 -> no ddp
ddp = int(os.environ.get('RANK', -1)) != -1
# if  now  is in ddp mode
if ddp:
    assert torch.cuda.is_available(), "CUDA required for DDP"
    # communication group
    init_process_group(backend='nccl')
    # global ranke
    ddp_rank = int(os.environ['RANK'])
    # local rank of  gpu of curent machine
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # # of gpus
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # tie process with each gpu
    # rank 0 → cuda:0
    # rank 1 → cuda:1
    # rank 2 → cuda:2
    # rank 3 → cuda:3
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # main process
    master_process = ddp_rank == 0

# do not support ddp
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else \
             'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# ============================================================
# 4. Training Loop
# ============================================================

total_batch_size = 524288
B, T = 64, 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated grad accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                              num_processes=ddp_world_size,
                              split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                            num_processes=ddp_world_size,
                            split="val")


torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304)).to(device)

# trick to accelerate
# compile the doc
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

optimizer = raw_model.configure_optimizers(0.1, 6e-4, device, master_process)


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


# gradient schedule
max_lr, min_lr, warmup_steps, max_steps = 6e-4, 6e-5, 715, 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# grad accum
# update: validation and hellaswag
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:

        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device,
                                    dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # training loop
    model.train()


    optimizer.zero_grad(set_to_none=True)
    loss_accum = torch.tensor(0.0, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} \
              | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms \
              | tok/sec: {tokens_per_sec:.2f}")
# ============================================================
# 5. Inference
# ============================================================
# cut process to release resources
if ddp: destroy_process_group()

model.eval()

# infer hyper params
num_return_sequences = 3
max_length = 40

enc = tiktoken.get_encoding('gpt2')
prefix = "Hello, I'm a language model,"

# tokenize using OpenAI tiktoken
tokens = torch.tensor(enc.encode(prefix), dtype=torch.long) \
            .unsqueeze(0).repeat(num_return_sequences, 1).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while tokens.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(tokens)
        # auto regression
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # keep first 50 words
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # random select
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        tokens = torch.cat((tokens, xcol), dim=1)

# tokens to sentences
for i in range(num_return_sequences):
    decoded = enc.decode(tokens[i, :max_length].tolist())
    print(f"> {decoded}")
