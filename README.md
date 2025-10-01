# gpt2-repro-124M
Reimplementation of GPT-2 (124M) from scratch in PyTorch, following Karpathy’s walkthrough.
---

## 📖 Introduction
This project aims to **reproduce GPT-2 (124M) from scratch** in PyTorch, following Andrej Karpathy’s walkthrough.  
Goal: deepen understanding of **Transformer architecture**, training optimization, and scaling behavior.
---
**Directory Structure (planned)**:
```

gpt2-repro/
├── src/          # Core model implementation
├── data/         # Dataset storage
├── scripts/      # Training / sampling scripts
├── results/      # Loss curves, sample generations
├── requirements.txt
└── README.md

````

---

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Acceleration**: CUDA / Mixed Precision
- **Tokenizer**: HuggingFace GPT-2 tokenizer

---

## How to Run
1. Clone the repo & install dependencies:
   ```bash
   git clone https://github.com/<your-username>/gpt2-repro.git
   cd gpt2-repro
   pip install -r requirements.txt
```

2. Train the model:

   ```bash
   python scripts/train.py
   ```
3. Generate text samples:

   ```bash
   python scripts/sample.py --prompt "Once upon a time"
   ```

---

## Results

* Training loss curves (to be added).
* Example generations (to be added).

---

## References

* [Karpathy’s video: *Let’s Reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU)
* [OpenAI GPT-2 Paper (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Karpathy’s build-nanogpt repo](https://github.com/karpathy/build-nanogpt)


