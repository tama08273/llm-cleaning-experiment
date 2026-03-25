import json
from .simple_tokenizer import WhitespaceTokenizer

def save_simple_vocab(tokenizer: WhitespaceTokenizer, path: str):
    """簡易トークナイザの語彙を JSON で保存"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.token_to_id, f, ensure_ascii=False, indent=2)

def load_simple_vocab(path: str) -> WhitespaceTokenizer:
    """簡易トークナイザの語彙を JSON から復元"""
    tok = WhitespaceTokenizer(vocab_size=5000, min_freq=1)
    with open(path, "r", encoding="utf-8") as f:
        tok.token_to_id = json.load(f)
    tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
    return tok