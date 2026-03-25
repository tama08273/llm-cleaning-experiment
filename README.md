# Decoder Transformer Mini (GPT-2 互換の自作実装)


本プロジェクトでは、LLMアーキテクチャの再現に加え、データクレンジングの影響評価を通じて、前処理設計の重要性を検証しています。

本リポジトリは、デコーダ専用 Transformer（GPT-2 互換アーキテクチャ）を自作実装し、  
学習・生成・前処理の比較実験を通じて、

> LLMの挙動理解および前処理設計がモデル性能に与える影響を検証する

ことを目的とした研究プロジェクトです。

本実装は OpenAI 公式とは無関係であり、商用利用は想定していません。

---

## 特徴

- 疎結合構成  
  トークナイザ、データローダ、モジュール、モデルを差し替えて比較検証しやすい

- 最小実装から拡張へ  
  GPT-2 互換の基本ブロック（因果マスク付き自己注意、Pre-LN、GELU、重み共有）をベースに拡張可能

- トークナイザ選択  
  簡易トークナイザ（空白区切り）と SentencePiece/BPE を切り替え可能

- クリーニング選択  
  前処理（cleaner）を切り替え、学習および生成挙動への影響を比較可能

- 実験志向設計  
  前処理・トークナイザ・データ条件を独立変数として扱い、同一条件で比較実験が可能

---

## リポジトリ構成
```
project-root/
├─ train.py              # 学習エントリーポイント
├─ generate.py           # 生成エントリーポイント（対話型・比較実験に使用）
├─ readme.md
├─ requirements.txt      # 依存パッケージ（例: torch, sentencepiece）
├─ myconfig/
│   ├─ basic_config.py         # モデルのハイパーパラメータ (GPT2Config)
│   └─ train_config.py   # 学習設定 (TrainConfig)
├─ tokenizer/
│   ├─ __init__.py
│   ├─ simple_tokenizer.py   # 空白区切りの簡易トークナイザ
│   ├─ sp_tokenizer.py       # SentencePiece/BPE トークナイザ
│   └─ vocab_io.py           # 簡易トークナイザ語彙の保存/読み込み
├─ modules/
│   ├─ __init__.py
│   └─ modules.py            # Attention, FFN, Block
├─ model/
│   ├─ __init__.py
│   └─ model.py              # GPT2LMHeadModel (+ generate 実装)
├─ data/
│   ├─ __init__.py
│   └─ data.py               # Dataset, DataLoader, collate
├─ cleaners/
│   ├─ __init__.py
│   ├─ basic_cleaner.py
│   └─ advanced_cleaner.py
├─ dataset/
│   ├─ base.txt
│   └─ raw_noisy.txt
├─ cleaned_data/             # クリーニング後データの出力先
└─ checkpoints/              # 学習済み重み出力先（.gitignore 推奨）

```

## 動作環境
- Python 3.9+
- PyTorch（CPU 可、GPU 推奨）
- SentencePiece を使う場合: pip install sentencepiece

requirements.txt に従ってインストールしてください:

` pip install -r requirements.txt `


## クイックスタート

1. データ生成

    ```bash
    python make_noisy_dataset.py
    ```
    - 青空文庫からコーパス生成
    - ノイズ混入（abc, 記号, 重複など）

<br>

2. 学習(3条件比較)

    ```bash
    python train.py --train_path dataset/raw_noisy.txt --cleaner none
    python train.py --train_path dataset/raw_noisy.txt --cleaner basic
    python train.py --train_path dataset/raw_noisy.txt --cleaner advanced
    ```


3. 生成比較

    ```bash
   python generate.py --ckpt checkpoints/gpt2mini_none.pt --vocab_json tokenizers/tokenizer_vocab_none.json
    python generate.py --ckpt checkpoints/gpt2mini_basic.pt --vocab_json tokenizers/tokenizer_vocab_basic.json
    python generate.py --ckpt checkpoints/gpt2mini_advanced.pt --vocab_json tokenizers/tokenizer_vocab_advanced.json
 
    ```

## データクレンジング実験

### 実験目的

LLMにおける前処理（データクレンジング）が

- 学習挙動
- 生成品質

に与える影響を検証する


### データ構築

青空文庫の複数作品を使用

- 走れメロス
- 羅生門
- こころ
- 坊っちゃん
- 注文の多い料理店
- 銀河鉄道の夜
- 人間失格

### ノイズ注入

- 英数字ノイズ（abc, 123）
- 重複文
- 記号列
- 文切断
- 短文ノイズ


### 比較条件

| 条件       | 内容                |
| -------- | ----------------- |
| none     | 前処理なし             |
| basic    | 空白正規化・制御文字除去      |
| advanced | ノイズ除去・短文フィルタ・重複除去 |


### 結果

| 条件       | 生成傾向            |
| -------- | --------------- |
| none     | ノイズ語（abc）が出力に反映 |
| basic    | 一部改善するがノイズ残存    |
| advanced | ノイズ減少するが出力が短くなる |

### 考察

- 前処理なし → ノイズがそのまま学習される
- 軽い前処理 → 不十分
- 強い前処理 → 生成性低下の可能性

### 結論

***<u>LLMにおける前処理は単なるノイズ除去ではなく、
ノイズ除去と生成性のバランス設計が重要である</u>***

### 拡張ポイント
- Tokenizer変更（BPE / SentencePiece）
- Attention改良（RoPE / ALiBi）
- 学習戦略変更（Optimizer / Scheduler）
- 生成制御（top-k / top-p / repetition_penalty）
- Cleanerの追加


### よくあるトラブル
- index out of range

    → vocab_size不一致

- UNK多発

    → 語彙不足 or データ不足

- 同語反復

    → temperature / top-k 調整

- 空出力

    → 前処理過剰 or データ短文化

### ハイパーパラメータ目安
- n_embd: 256〜768
- n_layer: 4〜12
- n_head: 4〜12
- vocab_size: 小規模なら 500〜5000


### 注意事項
- 本コードは研究・学習用途
- 商用利用非想定
- データライセンスは各自確認
- 大容量ファイルはGit管理対象外推奨


### 目的
- LLM内部理解
- 前処理設計の重要性理解
- 技術検証・研究用途