import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(text_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            # add_bos / add_eos は tokenizer 実装に合わせて調整
            ids = tokenizer.encode(line, add_bos=True, add_eos=True)

            # 2トークン未満だと shift できないので除外
            if len(ids) < 2:
                continue

            # 長すぎる場合は切る
            ids = ids[: self.max_length]

            # shift後に最低1トークン残ることを確認
            if len(ids) < 2:
                continue

            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch, pad_id=0):
    """
    GPT系の next token prediction 用:
      input_ids = seq[:-1]
      labels    = seq[1:]
    として1トークンずらす
    """
    batch_size = len(batch)
    max_len = max(len(seq) - 1 for seq in batch)  # shift後の長さ

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # ignore_index用

    for i, seq in enumerate(batch):
        x = seq[:-1]
        y = seq[1:]

        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        labels[i, : len(y)] = torch.tensor(y, dtype=torch.long)

    return input_ids, labels


def create_dataloader(text_path, tokenizer, batch_size=16, max_length=128, shuffle=True):
    dataset = TextDataset(text_path, tokenizer, max_length=max_length)

    # tokenizer に pad_id があれば使う。なければ 0
    pad_id = getattr(tokenizer, "pad_id", 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
    )