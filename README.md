# SmallGPT

A from-scratch implementation of GPT-style language models in PyTorch, with an interactive web playground for text generation. Character-level models trained on the Tiny Shakespeare dataset demonstrate the full transformer pipeline — from a simple bigram baseline to a multi-head attention decoder.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Architecture

### Bigram Model
The simplest language model — each character predicts the next using a single embedding lookup. No context window, no attention. Serves as a baseline to show what raw token-level prediction looks like before adding depth.

### Transformer Model
A decoder-only GPT with the full modern stack:

| Component | Detail |
|-----------|--------|
| Layers | 6 transformer blocks |
| Attention | 6-head causal self-attention |
| Embedding | 384-dim with learned positional encoding |
| Parameters | ~10.6M |
| Context | 256 characters |
| Normalization | Pre-norm LayerNorm |
| Regularization | Dropout (0.2) + residual connections |

## Project Structure

```
SmallGPT/
├── smallgpt/              # Core library
│   ├── config.py          # Hyperparameter dataclasses
│   ├── model.py           # Bigram + Transformer model definitions
│   └── tokenizer.py       # Character-level tokenizer
├── static/                # Web frontend
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   └── tiny_shakespeare.txt
├── train.py               # Training script with checkpoint saving
├── app.py                 # Flask API + web server
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
# Train the transformer (default)
python train.py

# Or train the simpler bigram model
python train.py --model bigram
```

Checkpoints are saved to `checkpoints/`.

### 3. Launch the web playground

```bash
python app.py
```

Open **http://localhost:5000** to generate text interactively. The web UI lets you:
- Pick between loaded models
- Adjust temperature, max tokens, and top-k sampling
- Enter custom prompts or generate unconditionally
- Copy output to clipboard

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List loaded models and their metadata |
| `POST` | `/api/generate` | Generate text from a model |
| `GET` | `/api/health` | Server health check |

### Generate request body

```json
{
    "model": "transformer",
    "prompt": "ROMEO:",
    "max_tokens": 500,
    "temperature": 0.8,
    "top_k": 40
}
```

## Sample Output

The trained transformer produces Shakespeare-like text:

```
ROMEO:
What say'st thou? Dost thou not hear me speak?
I am the master of my fortune's jest,
And shall not brook the idle tongue of fools
That prattle in the court of Verona.
```

*(Output quality depends on training duration and temperature settings.)*

## Training Notes

- **GPU recommended** for the transformer model (~10 min on a modern GPU, several hours on CPU)
- The bigram model trains in under a minute on CPU
- Adjust hyperparameters in `smallgpt/config.py`
- The training script prints loss metrics and saves a checkpoint when complete

## License

MIT
