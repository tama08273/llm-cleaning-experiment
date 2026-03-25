import argparse
import os
import torch

from myconfig.basic_config import GPT2Config
from model.model import GPT2LMHeadModel
from tokenizer import (
    WhitespaceTokenizer,
    SentencePieceTokenizer,
    load_simple_vocab,
)


def load_tokenizer_from_args(args):
    """
    保存済みトークナイザをロードする。
    - SentencePiece: spm.model をロード
    - simple: tokenizer_vocab_*.json をロード
    """
    if args.use_sp:
        if not os.path.exists(args.sp_model):
            raise FileNotFoundError(f"SentencePiece model not found: {args.sp_model}")
        tok = SentencePieceTokenizer()
        tok.load(args.sp_model)
        tok_type = "sentencepiece"
    else:
        if not os.path.exists(args.vocab_json):
            raise FileNotFoundError(f"Simple tokenizer vocab not found: {args.vocab_json}")
        # load_simple_vocab の実装に合わせる
        # 返り値が tokenizer の場合
        tok = load_simple_vocab(args.vocab_json)
        tok_type = "simple"
    return tok, tok_type


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg: GPT2Config = ckpt["config"]
    model = GPT2LMHeadModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=50,
    top_k=20,
    temperature=1.0,
    history=None,
):
    # 履歴は直近1往復だけに抑える
    if history:
        full_prompt = "\n".join(history[-2:] + [prompt])
    else:
        full_prompt = prompt

    input_ids = torch.tensor(
        [tokenizer.encode(full_prompt, add_bos=True, add_eos=False)],
        device=device
    )

    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=(None if top_k == 0 else top_k),
        temperature=temperature,
    )

    # 新規生成分だけ切り出す
    new_ids = out_ids[0][input_ids.shape[1]:].tolist()

    # decode の仕様に合わせて分岐
    try:
        return tokenizer.decode(new_ids, skip_special=True)
    except TypeError:
        return tokenizer.decode(new_ids)


def main():
    parser = argparse.ArgumentParser(description="Interactive generation using saved tokenizer.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--use_sp", action="store_true", help="Use SentencePiece tokenizer.")
    parser.add_argument("--sp_model", type=str, default="spm.model", help="Path to SentencePiece model file.")
    parser.add_argument("--vocab_json", type=str, required=False, default=None, help="Path to simple tokenizer vocab JSON.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling (0 to disable).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.use_sp and args.vocab_json is None:
        raise ValueError("When not using SentencePiece, --vocab_json is required.")

    tokenizer, tok_type = load_tokenizer_from_args(args)
    print(f"Loaded tokenizer ({tok_type}).")

    model, cfg, ckpt = load_model(args.ckpt, device)
    print(f"Loaded model from {args.ckpt} (vocab_size={cfg.vocab_size}).")

    print("=== 対話型生成を開始します ===")
    print("空行 または :q / :quit / :exit で終了します。")

    history = []
    while True:
        try:
            prompt = input("\n[あなた] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if prompt == "" or prompt.lower() in {":q", ":quit", ":exit"}:
            print("終了します。")
            break

        reply = generate_one(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            history=history,
        )

        history.append(prompt)
        history.append(reply)

        print("[モデル] >", reply)


if __name__ == "__main__":
    main()