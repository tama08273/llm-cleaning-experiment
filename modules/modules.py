import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from myconfig.basic_config import GPT2Config

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)

    def forward(self, position_ids):
        # position_ids: [B, T]
        return self.pos_emb(position_ids)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # QKV 一括
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        mask = torch.tril(torch.ones(config.n_positions, config.n_positions))
        self.register_buffer("causal_mask", mask.view(1, 1, config.n_positions, config.n_positions))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)                  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)         # 各 [B, T, C]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # [B, nh, T, T]
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                               # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        y = self.resid_drop(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc_in = nn.Linear(config.n_embd, hidden)
        self.fc_out = nn.Linear(hidden, config.n_embd)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-LN + 残差
        x = x + self.ffn(self.ln2(x))   # Pre-LN + 残差
        return x