import re

class advanced_cleaner():
    """
    - 青空文庫注記・ルビ除去
    - URL除去
    - 英数字ノイズ除去
    - 記号列圧縮
    - 連続文字圧縮
    - 短すぎる行や日本語をほぼ含まない行を除去
    - 重複除去
    """
    def __call__(self, lines):
        cleaned = []

        for line in lines:
            line = line.strip()
            line = line.replace("\u3000", " ")

            # 青空文庫系
            line = re.sub(r"［＃.*?］", "", line)
            line = re.sub(r"《.*?》", "", line)
            line = line.replace("｜", "")

            # URL
            line = re.sub(r"https?://\S+|www\.\S+", "", line)

            # 英数字ノイズ
            line = re.sub(r"\b[0-9A-Za-z]{2,}\b", "", line)

            # 記号列圧縮
            line = re.sub(r"[!！?？#＃・\.]{2,}", "", line)

            # 連続文字圧縮
            line = re.sub(r"(.)\1{3,}", r"\1\1", line)

            # 制御文字除去
            line = re.sub(r"[\u0000-\u001F\u007F]", "", line)

            # 空白正規化
            line = re.sub(r"\s+", " ", line).strip()

            # フィルタ
            if len(line) < 5:
                continue
            if not re.search(r"[ぁ-んァ-ン一-龠ー]", line):
                continue

            cleaned.append(line)

        # 重複除去
        cleaned = list(dict.fromkeys(cleaned))
        return cleaned