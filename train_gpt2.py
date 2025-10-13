# Import the dataclass decorator to easily create configuration classes

import math

# with default values and type hints.
from dataclasses import dataclass

# Import the main PyTorch library (used for tensors and core functions).
import torch

# Import the neural network module from PyTorch.
# This provides layers like Linear, Embedding, LayerNorm, etc.
import torch.nn as nn

# Import the functional API from torch.nn.
# This contains stateless functions (like activation functions, loss functions).
from torch.nn import functional as F

# --------------------------------------------------------- #

# =======================================
# Input x → Q, K, V projection → split into multiple heads → dot-product attention scores → causal mask (prevent looking ahead) → Softmax → weighted sum of values → concatenate heads → final Linear projection → output
# =======================================
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Make sure embedding size is divisible by number of heads
        assert config.n_embd % config.n_head == 0

        # Linear layer to project input embeddings into Q, K, V (query, key, value)
        # Instead of three separate matrices, we use one large projection and split later.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Linear projection for the output of attention (to mix heads back together)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # NOTE: follow the paper, scale using number of residual layers
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Store number of heads and embedding size
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask to prevent attending to future tokens
        # torch.tril creates a lower-triangular matrix (1 = allowed, 0 = masked).
        # register_buffer means it's stored with the model but not updated by gradients.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        # B = batch size, T = sequence length, C = embedding size
        B, T, C = x.size()

        # Project input x into q, k, v (queries, keys, values)
        qkv = self.c_attn(x)                          # Shape: (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)       # Split into (B, T, C) each

        # Reshape to separate attention heads
        # Shape: (B, nh, T, hs) where hs = head size = n_embd / n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        # # Compute raw attention scores: (Q x K^T) / sqrt(head_size)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        # # Apply causal mask: prevent attending to future positions
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # # Softmax over the last dimension (sequence length) to get probabilities
        # att = F.softmax(att, dim=-1)

        # # Weighted sum of values using the attention weights
        # y = att @ v   # Shape: (B, nh, T, hs)

        # Rearrange back: transpose and merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        y = self.c_proj(y)
        return y


# =======================================
# Define the MLP (feed-forward network) used inside each Transformer block.
# In GPT-style Transformers, this is typically a 2-layer fully connected network
# with a GELU activation in between.
# =======================================
class MLP(nn.Module):   # NOTE: should be nn.Module, not nn.module

    def __init__(self, config):
        # Initialize the parent nn.Module
        super().__init__()

        # First linear projection (c_fc):
        # Expands the hidden dimension from n_embd → 4 * n_embd.
        # This widening allows the model to capture more complex transformations.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # Activation function:
        # GELU (Gaussian Error Linear Unit) is used instead of ReLU.
        # The 'approximate="tanh"' version is slightly faster and commonly used.
        self.gelu = nn.GELU(approximate='tanh')

        # Second linear projection (c_proj):
        # Projects the expanded dimension back down: 4 * n_embd → n_embd.
        # This restores the hidden size so it can be added back to the residual stream.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # NOTE: follow the paper, scale using number of residual layers
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # Step 1: Project input up to 4 * n_embd
        x = self.c_fc(x)

        # Step 2: Apply GELU non-linearity (smooth activation)
        x = self.gelu(x)

        # Step 3: Project back down to n_embd
        x = self.c_proj(x)

        # Return the transformed output
        return x


# =======================================
# Define a single Transformer block.
# =======================================
# Each block consists of:
# 1. LayerNorm
# 2. Multi-head self-attention (causal, so it only looks at past tokens)
# 3. Another LayerNorm
# 4. Feed-forward MLP
# 5. Residual (skip) connections around attention and MLP
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        # First LayerNorm: applied before self-attention.
        # Helps stabilize training and normalize hidden states.
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # Multi-head causal self-attention module.
        # "Causal" means tokens cannot attend to future positions,
        # ensuring autoregressive prediction (next-token generation).
        self.attn = CausalSelfAttention(config)

        # Second LayerNorm: applied before the feed-forward MLP.
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Position-wise feed-forward network (MLP).
        # Usually: Linear → GELU → Linear
        # Expands and contracts the hidden dimension.
        self.mlp = MLP(config)

    def forward(self, x):
        # Apply LayerNorm → Attention, then add back the original input (residual connection).
        # This means: new_x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))

        # Apply LayerNorm → MLP, then again add residual connection.
        # new_x = x + MLP(LayerNorm(x))
        x = x + self.mlp(self.ln_2(x))

        # Return the updated hidden states.
        return x

# =======================================
# The @dataclass decorator automatically generates
# useful methods for the class (like __init__, __repr__).
# This makes it very convenient to define configuration objects
# with default values and type hints.
# =======================================
@dataclass
class GPTConfig:
    # Maximum context length (sequence length) the model can handle.
    # Example: 256 means the model can look at 256 tokens at once.
    block_size: int = 1024    # max sequence length (GPT-2 context window)

    # Size of the vocabulary (number of unique tokens/characters/words).
    # Each token ID in [0, vocab_size) maps to an embedding vector.
    vocab_size: int = 50257   # vocabulary size (BPE tokens)

    # Number of Transformer blocks (layers) to stack in the model.
    # Each block = self-attention + feed-forward + residual connections.
    n_layer: int = 12         # number of transformer layers

    # Number of attention heads inside each multi-head attention layer.
    # Each head learns different "attention patterns".
    n_head: int = 12          # number of attention heads

    # Dimensionality of the embedding vector for each token.
    # Also the hidden size of the model (width of the layers).
    n_embd: int = 768         # embedding/hidden dimension

# =========================================================
# Define the main GPT model class, which inherits from PyTorch's nn.Module.
# The skeleton of GPT2.
# tokens → embeddings + positions → stacked transformer blocks → normalization → linear head → logits
# =========================================================
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer core components
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: convert token IDs → embedding vectors
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embeddings: encode positions [0..block_size) → vectors
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Stack of Transformer blocks (n_layer deep)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final LayerNorm after all blocks
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Language modeling head: project hidden states back to vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # NOTE: fix bug
        # weight sharing scheme:
        # details: attention is all you need paper and 30-th refer
        # one way to do it
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        # Check if the module is a Linear layer (nn.Linear)
        if isinstance(module, nn.Linear):
            # Initialize weights from a normal distribution
            # mean = 0.0, std = 0.02
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= 2 * (self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # If the Linear layer has a bias term, set it to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # Check if the module is an Embedding layer (nn.Embedding)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights from the same normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        '''
        Input token IDs → convert to token embeddings
        Add position embeddings (so model knows order)
        Pass through stacked Transformer blocks (attention + MLP)
        Apply final LayerNorm
        Project to vocabulary logits (next-token prediction)
        '''
        # idx is the input sequence of token IDs
        # Shape: (B, T) where:
        #   B = batch size (number of sequences)
        #   T = sequence length (number of tokens in each sequence)
        B, T = idx.size()

        # Ensure sequence length does not exceed the maximum context length (block_size).
        # If T > block_size, the model cannot handle it and raises an error.
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # -------------------------------------------------------
        # 1. Compute position indices
        # torch.arange(0, T) creates [0, 1, 2, ..., T-1]
        # Shape: (T,)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # 2. Get position embeddings
        # Each position (0..T-1) maps to a learned embedding vector.
        # Shape: (T, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # 3. Get token embeddings
        # Each token ID in idx maps to its embedding vector.
        # Shape: (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 4. Add token embeddings + position embeddings
        # Broadcasting happens: pos_emb (T, n_embd) is added to each sequence in tok_emb.
        # Result shape: (B, T, n_embd)
        x = tok_emb + pos_emb

        # -------------------------------------------------------
        # 5. Pass through all Transformer blocks (stacked layers).
        # Each block applies: LayerNorm → Self-Attention → MLP → Residual connections
        for block in self.transformer.h:
            x = block(x)   # still (B, T, n_embd)

        # -------------------------------------------------------
        # 6. Final layer normalization before output
        # Shape: (B, T, n_embd)
        x = self.transformer.ln_f(x)


        # 7. Project hidden states to vocabulary logits
        # Linear layer maps each embedding (n_embd) → vocab_size
        # Shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        # -------------------------------------------------------
        # Return raw logits (before softmax).
        # Each position has a probability distribution over the vocabulary.
        return logits, loss


    # ===========================================================
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from HuggingFace checkpoints"""

        # ✅ Step 1: Ensure the requested model_type is valid
        # We only allow official GPT-2 variants (small, medium, large, xl).
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # Import HuggingFace's GPT-2 implementation
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # ✅ Step 2: Choose architecture hyperparameters based on model_type
        # Each GPT-2 variant has a different depth (n_layer), number of heads (n_head),
        # and hidden embedding size (n_embd).
        # These define the size of the neural network.
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # ~124M parameters
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # ~350M parameters
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # ~774M parameters
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # ~1558M parameters
        }[model_type]

        # ✅ Step 3: Add GPT-2 constants
        # Vocabulary size and context window size are always fixed for GPT-2 models.
        config_args['vocab_size'] = 50257   # GPT-2 BPE vocabulary size
        config_args['block_size'] = 1024    # Max sequence length (context size)

        # ✅ Step 4: Create our own GPT model (from scratch) with this config
        # This initializes random weights, not pretrained ones yet.
        config = GPTConfig(**config_args)
        model = GPT(config)

        # ✅ Step 5: Collect the parameter dictionary of our model
        # `state_dict` = all learnable parameters (weights, biases)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # Remove the attention bias mask from our keys
        # Because this buffer is not learnable (just a causal mask).
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # ------------------------------------------------------
        # ✅ Step 6: Load HuggingFace's pretrained GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # ✅ Step 7: Clean up HuggingFace parameter keys
        # They also have some buffer parameters we don’t care about.
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # ✅ Step 8: Define which weights need transposition
        # HuggingFace uses a custom "Conv1D" layer for efficiency,
        # while our implementation uses standard nn.Linear.
        # Conv1D stores weights as (out_features, in_features),
        # while nn.Linear expects (in_features, out_features).
        # → So we need to transpose these weights when copying.
        transposed = [
            'attn.c_attn.weight',   # Attention QKV projection
            'attn.c_proj.weight',   # Attention output projection
            'mlp.c_fc.weight',      # MLP expansion layer
            'mlp.c_proj.weight'     # MLP projection layer
        ]


        # Sanity check: number of parameters must match
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # Iterate over all parameters and copy weights
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special case: transpose Conv1D → Linear
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for normal parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


import tiktoken   # OpenAI's fast tokenizer library, here we use GPT-2 BPE encoding

class DataLoaderLite:
    def __init__(self, B, T):
        """
        Initialize the data loader.

        Args:
            B (int): batch size (number of sequences per batch)
            T (int): sequence length (number of tokens per sequence)
        """
        self.B = B
        self.T = T

        # ---------------------------------------------------------
        # 1. Load raw text from disk and encode it into tokens
        # ---------------------------------------------------------
        with open('input.txt', 'r') as f:
            text = f.read()   # read the whole file into a string
        enc = tiktoken.get_encoding('gpt2')  # use GPT-2 tokenizer
        tokens = enc.encode(text)            # convert text → list of token IDs
        self.tokens = torch.tensor(tokens)   # store as PyTorch tensor

        print(f"loaded {len(self.tokens)} tokens")  # total number of tokens in dataset
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        # Explanation:
        # - One "epoch" means processing the entire dataset once.
        # - Each batch consumes B*T tokens.
        # - So total number of batches per epoch = total_tokens // (B*T)

        # ---------------------------------------------------------
        # 2. Initialize state (where we are in the dataset)
        # ---------------------------------------------------------
        self.current_position = 0   # index pointer into the token tensor

    def next_batch(self):
        """
        Return the next batch of training data (x, y).
        Each batch is shaped:
            x: (B, T) input tokens
            y: (B, T) target tokens (shifted by 1)
        """
        B, T = self.B, self.T

        # ---------------------------------------------------------
        # 1. Slice a chunk of (B*T + 1) tokens from the dataset.
        # We need +1 because y is shifted by one token relative to x.
        # ---------------------------------------------------------
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        # ---------------------------------------------------------
        # 2. Create input (x) and target (y) sequences.
        # - x is all but the last token, reshaped into (B, T)
        # - y is all but the first token, reshaped into (B, T)
        # Example:
        #   buf = [t1, t2, t3, t4, t5]
        #   x   = [t1, t2, t3, t4]
        #   y   = [t2, t3, t4, t5]
        # ---------------------------------------------------------
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets

        # ---------------------------------------------------------
        # 3. Advance the current position by B*T tokens.
        # This means "move forward one batch".
        # ---------------------------------------------------------
        self.current_position += B * T

        # ---------------------------------------------------------
        # 4. If we've reached the end of the dataset,
        # reset back to the beginning for the next epoch.
        # ---------------------------------------------------------
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        # ---------------------------------------------------------
        # 5. Return a pair (x, y) for training.
        # ---------------------------------------------------------
        return x, y


# --------------------------------------------------------------
# Test: load small GPT-2 (124M) and check if it runs
import time
# attempt to autodetect the device
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

# Set seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get a data batch
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# # NOTE: be careful
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

train_loader = DataLoaderLite(B=16, T=1024)

# to see the difference
torch.set_float32_matmul_precision('high')
# get logits
model = GPT(GPTConfig(vocab_size=50304))

model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)

# ============================================================
# Optimization loop: trains the model parameters using AdamW
# ============================================================

# Create the optimizer.
# - AdamW = Adam optimizer with decoupled weight decay (better for transformers).
# - model.parameters() tells the optimizer which parameters to update.
# - lr = learning rate, 3e-4 means step size for each parameter update.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95)
                              ,eps=1e-8)

# Training loop for 50 iterations (steps).
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # --------------------------------------------------------
    # 1. Zero out (reset) the gradients from the previous step.
    # PyTorch accumulates gradients by default, so we must clear
    # them before each backward pass.
    optimizer.zero_grad()

    # --------------------------------------------------------
    # 2. Forward pass: feed input (x) and target (y) into the model.
    # The model returns:
    #   - logits: raw predictions of shape (B, T, vocab_size)
    #   - loss:   cross-entropy loss between logits and y
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    # --------------------------------------------------------
    # 3. Backward pass: compute gradients of the loss
    # with respect to all learnable parameters in the model.
    # This uses automatic differentiation.
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # --------------------------------------------------------
    # 4. Optimization step: update parameters using the computed gradients.
    # Each parameter θ is updated like:
    #   θ_new = θ_old - lr * gradient(θ)
    optimizer.step()

    # --------------------------------------------------------
    # 5. Print training progress.
    # .item() converts a PyTorch scalar tensor to a regular Python float.
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


print(loss)

# exit if needed
import sys; sys.exit(0)

 # --------------------------------------------------------
 # Inference
  # --------------------------------------------------------

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
