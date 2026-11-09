"""Flask API serving SmallGPT models for text generation."""

import os
import glob
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from smallgpt.tokenizer import CharTokenizer
from smallgpt.model import BigramLanguageModel, GPTLanguageModel

app = Flask(__name__, static_folder="static")
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
MODELS = {}


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    chars = ckpt["chars"]
    text_for_tokenizer = "".join(chars)
    tokenizer = CharTokenizer(text_for_tokenizer)

    if ckpt["model_type"] == "bigram":
        model = BigramLanguageModel(ckpt["vocab_size"])
    else:
        model = GPTLanguageModel(
            vocab_size=ckpt["vocab_size"],
            n_embd=ckpt["n_embd"],
            n_head=ckpt["n_head"],
            n_layer=ckpt["n_layer"],
            block_size=ckpt["block_size"],
            dropout=ckpt.get("dropout", 0.0),
            device=DEVICE,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer, ckpt["model_type"]


def discover_models():
    MODELS.clear()
    if not os.path.isdir(CHECKPOINT_DIR):
        return
    for path in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")):
        try:
            model, tokenizer, model_type = load_checkpoint(path)
            name = os.path.splitext(os.path.basename(path))[0]
            MODELS[name] = {"model": model, "tokenizer": tokenizer, "type": model_type}
            print(f"Loaded model: {name} ({model_type})")
        except Exception as e:
            print(f"Failed to load {path}: {e}")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/models", methods=["GET"])
def list_models():
    models = []
    for name, info in MODELS.items():
        param_count = sum(p.numel() for p in info["model"].parameters())
        models.append({
            "name": name,
            "type": info["type"],
            "parameters": f"{param_count / 1e6:.2f}M",
            "vocab_size": info["tokenizer"].vocab_size,
        })
    return jsonify({"models": models, "device": DEVICE})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model_name = data.get("model", "")
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 2000)
    temperature = max(0.01, min(data.get("temperature", 1.0), 2.0))
    top_k = data.get("top_k")

    if model_name not in MODELS:
        return jsonify({"error": f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"}), 404

    info = MODELS[model_name]
    model = info["model"]
    tokenizer = info["tokenizer"]

    if prompt:
        try:
            tokens = tokenizer.encode(prompt)
            context = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        except KeyError:
            return jsonify({"error": "Prompt contains characters not in the training vocabulary"}), 400
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    generated_text = tokenizer.decode(output[0].tolist())

    return jsonify({
        "text": generated_text,
        "prompt": prompt,
        "model": model_name,
        "tokens_generated": max_tokens,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": len(MODELS), "device": DEVICE})


if __name__ == "__main__":
    discover_models()
    print(f"Loaded {len(MODELS)} model(s) on {DEVICE}")
    app.run(host="0.0.0.0", port=5000, debug=True)
