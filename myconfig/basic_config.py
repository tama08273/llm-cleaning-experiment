from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int = 5000        # 語彙数（簡易トークナイザ用）
    n_positions: int = 256        # 最大系列長
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1