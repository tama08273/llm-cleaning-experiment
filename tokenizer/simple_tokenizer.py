import re
from collections import Counter

class WhitespaceTokenizer:
    """
    超簡易トークナイザ（空白区切り）。実用では BPE/SentencePiece に置換推奨。
    """
    def __init__(self, vocab_size=5000, lowercase=True, min_freq=2):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, texts):
        counter = Counter()
        for t in texts:
            if self.lowercase:
                t = t.lower()
            toks = re.findall(r"\S+", t)
            counter.update(toks)

        # special は除外し、頻度閾値を適用
        items = [
            (tok, c) for tok, c in counter.items()
            if c >= self.min_freq and tok not in self.special_tokens
        ]
        # 頻度順にソート
        items = sorted(items, key=lambda x: (-x[1], x[0]))

        # 重複を避けつつ、定義した vocab_size まで詰める
        seen = set(self.special_tokens)
        vocab_tokens = []
        for tok, _ in items:
            if tok in seen:
                continue
            vocab_tokens.append(tok)
            seen.add(tok)
            if len(vocab_tokens) >= self.vocab_size - len(self.special_tokens):
                break

        vocab = self.special_tokens + vocab_tokens
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text, add_bos=False, add_eos=True):
        if self.lowercase:
            text = text.lower()
        toks = re.findall(r"\S+", text)
        ids = []
        if add_bos:
            ids.append(self.token_to_id[self.bos_token])
        for tok in toks:
            ids.append(self.token_to_id.get(tok, self.token_to_id[self.unk_token]))
        if add_eos:
            ids.append(self.token_to_id[self.eos_token])
        return ids

    def decode(self, ids, skip_special=True):
        toks = []
        for i in ids:
            tok = self.id_to_token.get(i, self.unk_token)
            if skip_special and tok in self.special_tokens:
                continue
            toks.append(tok)
        return " ".join(toks)