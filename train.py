"""Train a character-level language model on text data."""

import argparse
import os
import json
import torch

from smallgpt.config import BigramConfig, TransformerConfig
from smallgpt.tokenizer import CharTokenizer
from smallgpt.model import BigramLanguageModel, GPTLanguageModel


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    text = load_data(args.data)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode_tensor(text)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    if args.model == "bigram":
        cfg = BigramConfig()
        model = BigramLanguageModel(tokenizer.vocab_size)
    else:
        cfg = TransformerConfig()
        model = GPTLanguageModel(
            vocab_size=tokenizer.vocab_size,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            device=device,
        )

    torch.manual_seed(cfg.seed)
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, cfg.block_size, cfg.batch_size, device, cfg.eval_iters
            )
            print(f"step {step:>5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(args.output, exist_ok=True)
    checkpoint = {
        "model_type": args.model,
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "chars": list(tokenizer._itos.values()),
    }
    if args.model == "transformer":
        checkpoint.update({
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "block_size": cfg.block_size,
            "dropout": cfg.dropout,
        })
    else:
        checkpoint["block_size"] = cfg.block_size

    path = os.path.join(args.output, f"{args.model}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)
    print("\n--- Sample Output ---")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmallGPT models")
    parser.add_argument("--model", choices=["bigram", "transformer"], default="transformer")
    parser.add_argument("--data", default="data/tiny_shakespeare.txt")
    parser.add_argument("--output", default="checkpoints")
    train(parser.parse_args())
