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

        # Compute raw attention scores: (Q x K^T) / sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        # Apply causal mask: prevent attending to future positions
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over the last dimension (sequence length) to get probabilities
        att = F.softmax(att, dim=-1)

        # Weighted sum of values using the attention weights
        y = att @ v   # Shape: (B, nh, T, hs)

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
        
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # hidden broadcasting
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    # ===========================================================
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from HuggingFace checkpoints"""
        
        # Only allow valid GPT-2 variants
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Different GPT-2 variants have different n_layer, n_head, n_embd
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        # vocab size and block size are always fixed for GPT-2
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Create our own GPT instance from scratch with the same config
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Grab its state_dict (all parameters)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Exclude attention bias buffer (not learnable, just a mask)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # ------------------------------------------------------
        # Load the HuggingFace model with pretrained weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # HuggingFace also has some buffer keys we can ignore
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # Some layers in HuggingFace are stored as Conv1D,
        # while we use plain Linear → need to transpose weights.
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
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

# --------------------------------------------------------------
# Test: load small GPT-2 (124M) and check if it runs
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")
