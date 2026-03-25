import random
import re
from pathlib import Path


# =========================
# 対象ファイル（7作品）
# =========================
SOURCE_FILES = [
    "bocchan.txt",
    "chumonno_oi_ryoriten.txt",
    "hashire_merosu.txt",
    "kokoro.txt",
    "rashomon.txt",
    "gingatetsudo.txt",      # 追加
    "ningen_shikkaku.txt",   # 追加
]

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_PATH = OUTPUT_DIR / "base.txt"
RAW_NOISY_PATH = OUTPUT_DIR / "raw_noisy.txt"


# =========================
# 青空文庫前処理
# =========================
def strip_aozora_artifacts(text: str) -> str:
    # ヘッダ・フッタざっくり削除
    text = re.sub(r"-------------------------------------------------------.*?-------------------------------------------------------", "", text, flags=re.DOTALL)

    # 注記・ルビ
    text = re.sub(r"［＃.*?］", "", text)
    text = re.sub(r"《.*?》", "", text)
    text = text.replace("｜", "")

    return text


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# =========================
# 行分割
# =========================
def split_into_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    out = []
    for line in lines:
        # 長文は句点で分割
        if len(line) > 120:
            parts = re.split(r"(?<=。)", line)
            parts = [p.strip() for p in parts if p.strip()]
            out.extend(parts)
        else:
            out.append(line)

    return out


# =========================
# エンコード自動判定
# =========================
def read_text_auto(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932")


# =========================
# baseデータ作成
# =========================
def build_base_dataset():
    all_lines = []

    for fname in SOURCE_FILES:
        path = INPUT_DIR / fname
        if not path.exists():
            print(f"skip: {path} not found")
            continue

        print(f"loading: {fname}")

        text = read_text_auto(path)
        text = strip_aozora_artifacts(text)
        text = normalize_text(text)
        lines = split_into_lines(text)

        # 短すぎる行除去
        lines = [line for line in lines if len(line) >= 5]

        print(f" -> {len(lines)} lines")

        all_lines.extend(lines)

    print(f"\nTOTAL base lines: {len(all_lines)}")
    return all_lines


# =========================
# ノイズ注入
# =========================
def inject_noise(lines, seed=42):
    random.seed(seed)
    noisy = list(lines)

    n = len(lines)

    # 1. 重複（5%）
    dup = random.sample(lines, min(n // 20, n))
    noisy.extend(dup)

    # 2. 記号ノイズ
    noisy.extend([
        "！！！！！！！！",
        "？？？？？？？？",
        "wwwwwwwwww",
        "########",
        "・・・・・・・・",
    ])

    # 3. URLノイズ
    noisy.extend([
        "http://example.com",
        "https://dummy.example.jp/test",
    ])

    # 4. 英数字混入
    for line in random.sample(lines, min(n // 30, n)):
        noisy.append(line + " 123 abc")

    # 5. 短文ノイズ
    noisy.extend([
        "テスト",
        "ああ",
        "NG",
        "OK",
    ])

    # 6. 文切断
    for line in random.sample(lines, min(n // 30, n)):
        if len(line) > 10:
            cut = random.randint(3, len(line) - 2)
            noisy.append(line[:cut])

    # 7. 連続文字
    for line in random.sample(lines, min(n // 30, n)):
        noisy.append(line + "あああああ")

    random.shuffle(noisy)

    print(f"TOTAL noisy lines: {len(noisy)}")
    return noisy


# =========================
# 保存
# =========================
def save_lines(lines, path):
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# =========================
# メイン
# =========================
def main():
    base_lines = build_base_dataset()
    save_lines(base_lines, BASE_PATH)

    noisy_lines = inject_noise(base_lines)
    save_lines(noisy_lines, RAW_NOISY_PATH)

    print("\n==== SAMPLE (base) ====")
    for line in base_lines[:5]:
        print(line)

    print("\n==== SAMPLE (noisy) ====")
    for line in noisy_lines[:10]:
        print(line)


if __name__ == "__main__":
    main()