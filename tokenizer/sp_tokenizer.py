# sp_tokenizer.py
import sentencepiece as spm
import os
from typing import List

class SentencePieceTokenizer:
    """
    SentencePieceベースのトークナイザ。
    - train() で .model / .vocab を生成
    - load() で既存モデルを読み込み
    - encode/decode で ID 列とテキストを相互変換
    """
    def __init__(self, model_prefix="spm", vocab_size=16000, character_coverage=0.9995):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.sp = None

    @property
    def pad_token(self): return "<pad>"
    @property
    def unk_token(self): return "<unk>"
    @property
    def bos_token(self): return "<bos>"
    @property
    def eos_token(self): return "<eos>"

    @property
    def pad_id(self): return self.sp.pad_id()
    @property
    def unk_id(self): return self.sp.unk_id()
    @property
    def bos_id(self): return self.sp.bos_id()
    @property
    def eos_id(self): return self.sp.eos_id()


    def train(self, input_text_path: str):
        # SentencePiece の学習。文字カバレッジは日本語向けに高め
        spm.SentencePieceTrainer.Train(
            input=input_text_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type="bpe",  # bpe/unigram など
            pad_id=0, pad_piece=self.pad_token,
            unk_id=1, unk_piece=self.unk_token,
            bos_id=2, bos_piece=self.bos_token,
            eos_id=3, eos_piece=self.eos_token,
        )
        self.load(f"{self.model_prefix}.model")

    def load(self, model_file: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file)

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special=True) -> str:
        if skip_special:
            ids = [i for i in ids if i not in {self.pad_id, self.unk_id, self.bos_id, self.eos_id}]
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size_(self):
        return self.sp.GetPieceSize()