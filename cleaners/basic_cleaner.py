import re

class basic_cleaner():
    """
    - 前後空白の除去
    - 重複空白の正規化
    - 制御文字の除去
    """
    def __call__(self, lines):
        out = []
        for line in lines:
            line = line.strip()
            # 制御文字除去
            line = re.sub(r"[\u0000-\u001F\u007F]", "", line)
            # 連続空白を1つに
            line = re.sub(r"\s+", " ", line)
            if line:
                out.append(line)
        return out
