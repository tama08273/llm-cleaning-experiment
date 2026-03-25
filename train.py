#ライブラリ
import argparse
import os
import torch
from pathlib import Path
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

#モジュールファイル

# モジュール
from myconfig.basic_config import GPT2Config
from myconfig.train_config import TrainConfig
from tokenizer import WhitespaceTokenizer, SentencePieceTokenizer #← トークナイザ差し替えポイント
from tokenizer import save_simple_vocab, load_simple_vocab
from data.data import create_dataloader # ← データローダ差し替えポイント
from model.model import GPT2LMHeadModel # ← モデル差し替えポイント（拡張ブロックを試す等）
from cleaners import get_cleaner  


def main():
    
    # --- CLI 引数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None, help="dataset path")
    parser.add_argument("--cleaner", type=str, default=None, help="none/basic/url_emoji")
    parser.add_argument("--use_sp", action="store_true", help="Use SentencePiece tokenizer")
    parser.add_argument("--sp_model", type=str, default="spm.model", help="Path to SentencePiece model")
    parser.add_argument("--sp_vocab_size", type=int, default=16000, help="SentencePiece vocab size (when training)")
    args = parser.parse_args()

    # --- 設定読み込み（デフォルト） ---
    cfg_train = TrainConfig()

    # --- CLI があれば優先 ---
    train_path = args.train_path or cfg_train.train_path
    cleaner_name = (args.cleaner or getattr(cfg_train, "cleaner", "none")).lower()
    use_sp = args.use_sp or getattr(cfg_train, "use_sp", False)
    sp_model_path = args.sp_model
    sp_vocab_size = args.sp_vocab_size

    print(f"args.cleaner={args.cleaner}, cleaner_name={cleaner_name}, cfg_train.cleaner={getattr(cfg_train,'cleaner','(なし)')}")

    epochs = cfg_train.epochs
    batch_size = cfg_train.batch_size
    max_length = cfg_train.max_length
    lr = cfg_train.lr
    weight_decay = cfg_train.weight_decay
    grad_clip = cfg_train.grad_clip
    device = "cuda" if (cfg_train.use_cuda and torch.cuda.is_available()) else "cpu"    
    
    # ---- テキスト読込 ----
    with open(train_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"raw lines: {len(texts)}")


    # ---- クリーニング（選択制）----
    # ---- クリーニング（選択制）----

    cleaner = get_cleaner(cleaner_name)
    if cleaner:
        texts = cleaner(texts)
        st = f"{cleaner_name} クリーニング適用後: {len(texts)} 行"
    else:
        st = f"クリーニングなし: {len(texts)} 行"
    print(st)

    # ---- クリーニング済みデータを保存----
    cleaned_dir = Path("cleaned_data")
    cleaned_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    cleaned_path = cleaned_dir / f"{Path(train_path).stem}_{cleaner_name}_{ts}.txt"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")
    print(f"cleaned data saved to: {cleaned_path}")   


    # ==== トークナイザ部（差し替えポイント） ====
    if use_sp:
        tokenizer = SentencePieceTokenizer(model_prefix="spm", vocab_size=sp_vocab_size)
        if os.path.exists(sp_model_path):
            tokenizer.load(sp_model_path)
            print(f"Loaded SentencePiece model: {sp_model_path}")
        else:
            print(f"Training SentencePiece model (vocab_size={sp_vocab_size})...")
            tokenizer.train(str(cleaned_path))
            print("SentencePiece model trained and saved (spm.model / spm.vocab).")
        vocab_size = tokenizer.vocab_size_   
    else:
        tokenizer = WhitespaceTokenizer(vocab_size=5000, min_freq=1)
        tokenizer.train(texts)
        vocab_size = len(tokenizer.token_to_id)
        os.makedirs("tokenizers", exist_ok=True)
        tokenizer_save_path = f"tokenizers/tokenizer_vocab_{cleaner_name}.json"
        save_simple_vocab(tokenizer, tokenizer_save_path)
        print(f"Saved simple tokenizer vocab to {tokenizer_save_path}")

    # ---- DataLoader（クリーニング後ファイルを使用）----
    train_loader = create_dataloader(cleaned_path, tokenizer, batch_size, max_length, shuffle=True) 

    # ==== モデル定義部（差し替えポイント） ====
    # GPT2Config のハイパーパラメータを変えることで層数/ヘッド数/埋め込み次元/コンテキスト長を変更できる。
    # また、GPT2LMHeadModel を拡張した別クラスに差し替えれば、RoPE などの位置表現やMoEなどを試せる。
    
    # ==== モデル定義 ====
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    print(model)

    # デバッグ用チェック（語彙外IDがないか確認）
    batch = next(iter(train_loader))
    input_ids, _ = batch
    #print("vocab_size (len):", len(vocab_size))
    print("max token id    :", input_ids.max().item())
    #assert input_ids.max().item() < len(tokenizer.token_to_id), "ID が語彙サイズを超えています"


    # ==== 最適化・スケジューラ部（差し替えポイント） ====
    # optimizer/scheduler を変更することで学習挙動を検証できる（例: Adafactor, OneCycleLRなど）。
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))


    # ==== 学習ループ ====
    batch = next(iter(train_loader))
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits, loss = model(input_ids, labels=labels)

    print("debug loss:", loss.item())
    print("logits shape:", logits.shape)
    print("logits min :", logits.min().item())
    print("logits max :", logits.max().item())
    print("logits mean:", logits.mean().item())
    print("logits std :", logits.std().item())
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, (input_ids, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, loss = model(input_ids, labels=labels)
            loss.backward()

            # 勾配クリッピング（発散防止）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 50 == 0:
                avg = total_loss / step
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {avg:.4f}")

        print(f"[epoch {epoch}] mean loss: {total_loss / len(train_loader):.4f}")



    # ==== 生成デモ ====
    # 生成部を差し替えることで top-p や温度制御、バッチ生成などを試せる。
    prompt = "こんにちは 世界"
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], device=device)
    generated = model.generate(input_ids, max_new_tokens=20, top_k=20)
    text_out = tokenizer.decode(generated[0].tolist())
    print("=== generation ===")
    print(text_out)


    # ==== モデル保存 ====
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/gpt2mini_{cleaner_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "cleaner": cleaner_name,
            "tokenizer_path": tokenizer_save_path if not use_sp else sp_model_path,
            "use_sp": use_sp,
        },
        ckpt_path,
    )
    print(f"saved to {ckpt_path}")
    
    batch = next(iter(train_loader))
    input_ids, labels = batch

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    print("input_ids sample:", input_ids[0][:20].tolist())
    print("labels sample   :", labels[0][:20].tolist())

    print("labels max:", labels.max().item())
    print("labels min:", labels.min().item())
    
    print("tokenizer vocab size:", vocab_size)
    print("model vocab size:", cfg.vocab_size)


if __name__ == "__main__":
    main()
