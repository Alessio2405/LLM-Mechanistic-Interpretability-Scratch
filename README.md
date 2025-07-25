# LLM & Mechanistic Interpretability from Scratch

My journey with https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch this course about basic to advanced topics like Transformers, full LLM, Mechanistic Interpretability etc.

## Quick Start

1. **Clone & install**

   ```bash
   git clone [https://github.com/Alessio2405/Trasformer-From-Scratch](https://github.com/Alessio2405/LLM-Mechanistic-Interpretability-Scratch)
   cd transformer-from-scratch
   pip install -r requirements.txt
   ```
2. **Run tests**

   ```bash
   pytest -q
   ```
3. **Try a forward pass**

   ```python
   from demo_transformer import DemoTransformer
   from config import Config
   import torch

   cfg = Config()
   model = DemoTransformer(cfg)
   tokens = torch.randint(0, cfg.d_vocab, (1, 10))
   logits = model(tokens)
   print(logits.shape)  # e.g., (1, 10, 50257)
   ```

## Configuration

Edit `Config` in `config.py` to tweak:

* `d_model`, `n_heads`, `d_mlp`, `n_layers`, etc.

## License

MIT
