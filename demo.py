#!/usr/bin/env python3
"""
demo.py - Interactive CLI Keyboard Simulator for Greek Keyboard LM

Simulates the real mobile keyboard UI:
1. Type a sentence prefix (e.g. "Καλημέρα, τι ") -> Shows the 3-word Keyboard Suggestion Bar + Top-5 next words.
2. Type an unaccented or typo word (e.g. "καλημερα") -> Automatically runs <XBU>...<XBC> autocorrect.
3. Type "q", "exit", or "quit" to exit.
"""

import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path
from transformers import LlamaForCausalLM


def load_engine(model_dir: str = "models/checkpoints/best_model", tokenizer_path: str = "models/tokenizer/tokenizer.model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[1;34m[-] Loading model from {model_dir} on {device}...\033[0m")
    model = LlamaForCausalLM.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"\033[1;34m[-] Loading SentencePiece tokenizer from {tokenizer_path}...\033[0m")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    return model, sp, device


def predict_next_words(model, sp, device, text: str, top_k: int = 5):
    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    token_ids = [bos_id] + sp.EncodeAsIds(text)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor).logits[0, -1, :]
        # Suppress special tokens (unk, bos, eos, pad, <XBU>, <XBC>, <XEC>)
        for special_id in [0, 1, 2, 3, 4, 5, 6]:
            logits[special_id] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=top_k)

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        piece = sp.IdToPiece(idx.item())
        clean_piece = piece.replace(" ", " ")
        predictions.append((clean_piece, prob.item(), idx.item()))
    return predictions


def autocorrect_word(model, sp, device, word: str, context: str = ""):
    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    xec_id = sp.PieceToId("<XEC>")
    xbc_id = sp.PieceToId("<XBC>")

    if context:
        prompt_text = f"{context.strip()} <XBU>{word.strip()}<XBC>"
    else:
        prompt_text = f"<XBU>{word.strip()}<XBC>"

    token_ids = [bos_id] + sp.EncodeAsIds(prompt_text)

    curr_input = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated_ids = []
    with torch.no_grad():
        for _ in range(8):
            logits = model(curr_input).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            if next_token == xec_id or next_token == sp.eos_id():
                break
            generated_ids.append(next_token)
            curr_input = torch.cat([curr_input, torch.tensor([[next_token]], device=device)], dim=1)

    return sp.DecodeIds(generated_ids).strip()


def run_interactive():
    model_dir = "models/checkpoints/best_model/final_model"
    tokenizer_path = "models/tokenizer/tokenizer.model"

    if not Path(model_dir).exists():
        model_dir = "models/checkpoints/best_model/best_model"
        print(f"\033[1;31m[!] Model or tokenizer not found at {model_dir}. Please ensure checkpoints exist.\033[0m")
        sys.exit(1)

    model, sp, device = load_engine(model_dir, tokenizer_path)

    print("\n" + "=" * 65)
    print("\033[1;32m 📱 GREEK KEYBOARD LM - INTERACTIVE TESTING CONSOLE 📱\033[0m")
    print("=" * 65)
    print("Instructions:")
    print("  • Type any Greek text/prefix: e.g. \033[1;33m'Καλημέρα, τι '\033[0m or \033[1;33m'θα ήθελα να '\033[0m")
    print("  • Type an unaccented word:    e.g. \033[1;33m'καλημερα'\033[0m or \033[1;33m'ειμαι'\033[0m")
    print("  • Type \033[1;31m'exit'\033[0m or \033[1;31m'quit'\033[0m to leave")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("\033[1;36m[Keyboard Input] > \033[0m").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                print("\033[1;34mGoodbye!\033[0m")
                break

            # 1. Next-Token Predictions
            preds = predict_next_words(model, sp, device, user_input, top_k=5)

            # Mobile Keyboard 3-Candidate Suggestion Bar
            cand1 = preds[0][0].strip() if len(preds) > 0 else ""
            cand2 = preds[1][0].strip() if len(preds) > 1 else ""
            cand3 = preds[2][0].strip() if len(preds) > 2 else ""

            print("\n  ┌─────────────────────────────────────────────────────────────┐")
            print(f"  │ 📱 Keyboard Bar: [ \033[1;32m{cand1:<10}\033[0m ] [ \033[1;33m{cand2:<10}\033[0m ] [ \033[1;34m{cand3:<10}\033[0m ] │")
            print("  └─────────────────────────────────────────────────────────────┘")

            print("  \033[1;37mTop 5 Next-Word Candidates:\033[0m")
            for rank, (piece, prob, tid) in enumerate(preds, start=1):
                bar = "█" * int(prob * 30)
                print(f"    {rank}. \033[1;33m{piece:<15}\033[0m (Prob: {prob*100:5.2f}%)  {bar}")

            # 2. If user typed a single word, also test autocorrect
            words = user_input.split()
            if len(words) == 1:
                ac_result = autocorrect_word(model, sp, device, words[0])
                if ac_result:
                    print(f"\n  \033[1;35m⚡ Autocorrect `<XBU>{words[0]}<XBC>` ➔ '{ac_result}'\033[0m")

            print("\n" + "-" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\n\033[1;34mExiting interactive console...\033[0m")
            break


if __name__ == "__main__":
    run_interactive()
