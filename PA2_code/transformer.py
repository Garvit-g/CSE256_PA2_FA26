"""Transformer encoder (Part 1) and decoder (Part 2), implemented from scratch.

Only basic PyTorch layers are used (nn.Linear, nn.Embedding, nn.LayerNorm); attention, the transformer
blocks and both models are written by hand. Every model returns its attention maps so they can be
checked and plotted with utilities.py: one map of shape [batch, T, T] per head per layer, in the order
layer 1 head 1, layer 1 head 2, ..., layer N head H.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0  # index of <pad> in SimpleTokenizer


class AttentionHead(nn.Module):
    """One head of scaled dot-product self-attention."""

    def __init__(self, n_embd, head_size, causal):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal

    def forward(self, x, pad_mask=None):
        # x: [B, T, C]; pad_mask: [B, T], True where the token is padding (those keys are ignored)
        T = x.shape[1]
        q, k, v = self.query(x), self.key(x), self.value(x)               # [B, T, hs]
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])         # [B, T, T]: row = query, column = key
        if self.causal:
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, float("-inf"))            # position i only sees keys j <= i
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)                                  # each row sums to 1
        return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel; their outputs are concatenated and projected back to n_embd."""

    def __init__(self, n_embd, n_head, causal):
        super().__init__()
        assert n_embd % n_head == 0
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_head, causal) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, pad_mask=None):
        outs, maps = zip(*(head(x, pad_mask) for head in self.heads))
        return self.proj(torch.cat(outs, dim=-1)), list(maps)


class FeedForward(nn.Module):
    """Position-wise feedforward layer: n_embd -> ff_hidden -> n_embd with ReLU."""

    def __init__(self, n_embd, ff_hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: self-attention then feedforward, each with a residual connection (pre-LayerNorm)."""

    def __init__(self, n_embd, n_head, ff_hidden, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, causal)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, ff_hidden)

    def forward(self, x, pad_mask=None):
        a, maps = self.attn(self.ln1(x), pad_mask)
        x = x + a
        x = x + self.ffwd(self.ln2(x))
        return x, maps


class TokenAndPositionEmbedding(nn.Module):
    """Learned absolute position embeddings: one table for tokens, one for positions, added together."""

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        T = idx.shape[1]
        return self.tok(idx) + self.pos(torch.arange(T, device=idx.device))


def _init_weights(module):
    """GPT-2 style initialization: N(0, 0.02) weights, zero biases (init="gpt2", the default).
    With init="default" the PyTorch defaults are kept: N(0, 1) embeddings, Kaiming-uniform linear layers."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class TransformerEncoder(nn.Module):
    """Bidirectional encoder: every position attends to every non-padding position.
    Returns one contextual vector per input token, plus the attention maps."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, ff_hidden, mask_padding=True, init="gpt2"):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, ff_hidden, causal=False) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.mask_padding = mask_padding
        if init == "gpt2":
            self.apply(_init_weights)

    def forward(self, idx):
        pad_mask = (idx == PAD_ID) if self.mask_padding else None
        x = self.embed(idx)
        maps = []
        for block in self.blocks:
            x, m = block(x, pad_mask)
            maps += m
        return self.ln_f(x), maps                                         # [B, T, n_embd], list of [B, T, T]


class FeedForwardClassifier(nn.Module):
    """One hidden layer: n_input -> n_hidden (ReLU) -> n_output scores."""

    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return self.net(x)


class EncoderClassifier(nn.Module):
    """Encoder + mean over the sequence + feedforward classifier, trained jointly end to end."""

    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, idx):
        h, maps = self.encoder(idx)
        if self.encoder.mask_padding:                                     # mean over the real (non-padding) tokens
            keep = (idx != PAD_ID).unsqueeze(-1).float()
            mean = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        else:
            mean = h.mean(dim=1)
        return self.classifier(mean), maps                                # logits [B, n_output]


class TransformerDecoder(nn.Module):
    """GPT-like decoder: masked (causal) self-attention, so position i only sees positions <= i.
    forward(idx) returns (logits, attention maps); forward(idx, targets) returns (mean cross-entropy loss, maps)."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, ff_hidden=100, init="gpt2"):
        super().__init__()
        self.embed = TokenAndPositionEmbedding(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, ff_hidden, causal=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if init == "gpt2":
            self.apply(_init_weights)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        maps = []
        for block in self.blocks:
            x, m = block(x)
            maps += m
        logits = self.lm_head(self.ln_f(x))                               # [B, T, vocab]
        if targets is None:
            return logits, maps
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, maps


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
