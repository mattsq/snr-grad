"""Small decoder-only Transformer LM and a synthetic Markov language (for E7)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q = q.view(b, l, h, d // h).transpose(1, 2)
        k = k.view(b, l, h, d // h).transpose(1, 2)
        v = v.view(b, l, h, d // h).transpose(1, 2)
        # Explicit attention (forward-mode AD friendly, unlike fused SDPA kernels).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // h)
        mask = torch.ones(l, l, dtype=torch.bool, device=x.device).triu(1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, l, d)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        l = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(l, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


class MarkovLanguage:
    """Learnable synthetic language: order-1 Markov chain plus a copy rule.

    Each next token follows a Zipf-like, temperature-sharpened transition from the
    previous token; with probability ``copy_prob`` it instead repeats the token
    ``copy_lag`` positions back (learnable only through attention).
    """

    def __init__(self, vocab: int = 64, temperature: float = 0.5, copy_prob: float = 0.3,
                 copy_lag: int = 8, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        ranks = torch.arange(1, vocab + 1, dtype=torch.float32)
        base = -torch.log(ranks)
        perm = torch.argsort(torch.rand(vocab, vocab, generator=g), dim=-1)
        logits = base[perm] + 0.3 * torch.randn(vocab, vocab, generator=g)
        self.probs = torch.softmax(logits / temperature, dim=-1)  # [prev, next]
        self.vocab = vocab
        self.copy_prob = copy_prob
        self.copy_lag = copy_lag

    def sample(self, n: int, length: int, gen: torch.Generator) -> torch.Tensor:
        x = torch.empty(n, length + 1, dtype=torch.long)
        x[:, 0] = torch.randint(0, self.vocab, (n,), generator=gen)
        for t in range(1, length + 1):
            nxt = torch.multinomial(self.probs[x[:, t - 1]], 1, generator=gen)[:, 0]
            if t >= self.copy_lag:
                copy = torch.rand(n, generator=gen) < self.copy_prob
                nxt = torch.where(copy, x[:, t - self.copy_lag], nxt)
            x[:, t] = nxt
        return x

    def entropy_floor(self) -> float:
        """Rough lower bound on achievable loss (Markov part only, nats)."""
        p = self.probs
        h = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
        return float((1 - self.copy_prob) * h)


def lm_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))


def output_loss_fn(batch: torch.Tensor):
    """Output-space loss for GGNCurvature: logits -> mean cross-entropy."""
    targets = batch[:, 1:].reshape(-1)

    def f(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)

    return f
