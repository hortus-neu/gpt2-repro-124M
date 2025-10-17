import torch, tiktoken
from train_gpt2 import GPT, GPTConfig

ckpt = torch.load("log/model_19073.pt", map_location="cuda")
model = GPT(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.eval().cuda()

enc = tiktoken.get_encoding("gpt2")
prefix = "Hello, I'm a language model,"
tokens = torch.tensor(enc.encode(prefix), dtype=torch.long)[None, :].cuda()

max_length = 64
for _ in range(max_length - tokens.size(1)):
    with torch.no_grad():
        logits, _ = model(tokens)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

print(enc.decode(tokens[0].tolist()))
