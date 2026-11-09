from dataclasses import dataclass


@dataclass
class BigramConfig:
    batch_size: int = 32
    block_size: int = 8
    max_iters: int = 3000
    eval_interval: int = 300
    learning_rate: float = 1e-2
    eval_iters: int = 200
    seed: int = 1337


@dataclass
class TransformerConfig:
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    seed: int = 1337
