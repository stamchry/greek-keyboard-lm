#!/usr/bin/env python3
"""
07_evaluate_model.py - Comprehensive Evaluation & Interactive Benchmark Suite

Evaluates:
1. Next-Token Perplexity and Top-k Accuracy (Top-1, Top-3, Top-5, Top-10) on test split.
2. Accent Restoration Accuracy (>95% target) on 1,000 unaccented Greek words.
3. Synthetic Typo Autocorrect Accuracy (keyboard adjacency, homophones, transpositions).
4. Interactive Terminal Mode for real-time testing of suggestions and autocorrect.
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import sentencepiece as spm
from transformers import LlamaForCausalLM, LlamaConfig

# Import GreekCorrupter from 04_generate_corruptions
try:
    from importlib.machinery import SourceFileLoader
    corruptions_module = SourceFileLoader(
        "generate_corruptions",
        str(Path(__file__).parent / "04_generate_corruptions.py")
    ).load_module()
    GreekCorrupter = corruptions_module.GreekCorrupter
except Exception:
    GreekCorrupter = None


def load_model_and_tokenizer(model_dir: Path, tokenizer_file: Path, device: torch.device):
    """Load model and SentencePiece processor."""
    print(f"[-] Loading model from {model_dir}...")
    model = LlamaForCausalLM.from_pretrained(str(model_dir)).to(device)
    model.eval()

    print(f"[-] Loading tokenizer from {tokenizer_file}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_file))

    return model, sp


def evaluate_perplexity_and_topk(
    model: LlamaForCausalLM,
    sp: spm.SentencePieceProcessor,
    test_file: Path,
    device: torch.device,
    max_samples: int = 500
) -> Dict[str, float]:
    """Compute perplexity and Top-1, Top-3, Top-5, Top-10 accuracy on test text."""
    if not test_file.exists():
        print(f"[!] Test file not found at {test_file}. Skipping test perplexity.")
        return {}

    lines = []
    with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    if len(lines) > max_samples:
        lines = lines[:max_samples]

    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    eos_id = sp.eos_id() if sp.eos_id() != -1 else 2

    total_nll = 0.0
    total_tokens = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0

    print(f"[-] Evaluating Next-Token Perplexity and Top-K on {len(lines)} test sentences...")

    with torch.no_grad():
        for line in lines:
            token_ids = [bos_id] + sp.EncodeAsIds(line) + [eos_id]
            if len(token_ids) < 2:
                continue

            input_tensor = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
            target_tensor = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

            logits = model(input_tensor).logits  # [1, seq_len, vocab_size]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_tensor.view(-1), reduction='sum')

            total_nll += loss.item()
            total_tokens += target_tensor.numel()

            # Top-K accuracy
            # logits: [1, seq_len, vocab]
            _, top10_preds = torch.topk(logits, k=10, dim=-1)  # [1, seq_len, 10]
            targets_expanded = target_tensor.unsqueeze(-1)  # [1, seq_len, 1]

            top1_correct += (top10_preds[:, :, :1] == targets_expanded).sum().item()
            top3_correct += (top10_preds[:, :, :3] == targets_expanded).sum().item()
            top5_correct += (top10_preds[:, :, :5] == targets_expanded).sum().item()
            top10_correct += (top10_preds[:, :, :10] == targets_expanded).sum().item()

    avg_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))

    results = {
        "test_tokens": total_tokens,
        "cross_entropy_loss": avg_loss,
        "perplexity": ppl,
        "top1_acc": top1_correct / max(1, total_tokens),
        "top3_acc": top3_correct / max(1, total_tokens),
        "top5_acc": top5_correct / max(1, total_tokens),
        "top10_acc": top10_correct / max(1, total_tokens),
    }
    return results


def evaluate_accent_restoration(
    model: LlamaForCausalLM,
    sp: spm.SentencePieceProcessor,
    test_file: Path,
    device: torch.device,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Evaluates accent restoration accuracy on unaccented Greek words:
    Format: [Context] <XBU>[unaccented_word]<XBC> -> model should generate [accented_word]<XEC>
    """
    print(f"[-] Running Accent Restoration Benchmark ({num_samples} samples)...")

    if GreekCorrupter is None:
        print("[!] GreekCorrupter not available. Skipping accent test.")
        return {}

    corrupter = GreekCorrupter(seed=999)

    # Collect polysyllabic accented words from test file
    candidate_words = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for w in line.split():
                    clean_w = w.strip(".,;:!?\"'()«»[]{} -—")
                    if corrupter.has_accents(clean_w) and len(clean_w) >= 3:
                        candidate_words.append(clean_w)

    if not candidate_words:
        # Fallback list of common Greek accented words
        candidate_words = [
            "καλημέρα", "είμαι", "θέλω", "παιδιά", "πάμε", "άνθρωπος",
            "σπίτι", "βιβλίο", "τρένο", "χρόνος", "ημέρα", "όμορφος",
            "πολύ", "ευχαριστώ", "παρακαλώ", "ερώτηση", "απάντηση"
        ] * (num_samples // 10 + 1)

    import random
    random.seed(999)
    random.shuffle(candidate_words)
    candidate_words = candidate_words[:num_samples]

    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    xec_id = sp.PieceToId("<XEC>")
    xbc_id = sp.PieceToId("<XBC>")

    correct_top1 = 0
    correct_top3 = 0
    total_evaluated = 0

    with torch.no_grad():
        for target_word in candidate_words:
            unaccented = corrupter.strip_accents(target_word)
            if unaccented == target_word:
                continue

            prompt_text = f"<XBU>{unaccented}<XBC>"
            prompt_ids = [bos_id] + sp.EncodeAsIds(prompt_text)

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Generate greedily up to max 10 tokens
            generated_ids = []
            curr_input = input_tensor

            for _ in range(8):
                logits = model(curr_input).logits[:, -1, :]  # [1, vocab]
                next_token = torch.argmax(logits, dim=-1).item()
                if next_token == xec_id or next_token == sp.eos_id():
                    break
                generated_ids.append(next_token)
                curr_input = torch.cat([curr_input, torch.tensor([[next_token]], device=device)], dim=1)

            predicted_text = sp.DecodeIds(generated_ids).strip()
            total_evaluated += 1

            if predicted_text == target_word:
                correct_top1 += 1
                correct_top3 += 1
            elif predicted_text.lower() == target_word.lower():
                correct_top1 += 1
                correct_top3 += 1

    acc = correct_top1 / max(1, total_evaluated)
    return {
        "samples_evaluated": total_evaluated,
        "accent_accuracy": acc,
        "target_met": acc >= 0.95
    }


def evaluate_synthetic_autocorrect(
    model: LlamaForCausalLM,
    sp: spm.SentencePieceProcessor,
    eval_jsonl_file: Path,
    device: torch.device,
    max_samples: int = 500
) -> Dict[str, float]:
    """Evaluate full autocorrect suite on JSONL benchmark file."""
    if not eval_jsonl_file.exists():
        print(f"[!] Evaluation file {eval_jsonl_file} not found. Skipping autocorrect benchmark.")
        return {}

    samples = []
    with open(eval_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    if len(samples) > max_samples:
        samples = samples[:max_samples]

    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    xec_id = sp.PieceToId("<XEC>")
    xbc_id = sp.PieceToId("<XBC>")

    correct = 0
    total = 0
    by_type = {}

    with torch.no_grad():
        for sample in samples:
            context = sample.get("context", "")
            corrupted = sample.get("corrupted_word", "")
            target = sample.get("target_word", "")
            err_type = sample.get("error_type", "unknown")

            if err_type not in by_type:
                by_type[err_type] = {"total": 0, "correct": 0}

            if context:
                prompt_text = f"{context} <XBU>{corrupted}<XBC>"
            else:
                prompt_text = f"<XBU>{corrupted}<XBC>"

            prompt_ids = [bos_id] + sp.EncodeAsIds(prompt_text)
            curr_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            generated_ids = []
            for _ in range(8):
                logits = model(curr_input).logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).item()
                if next_token == xec_id or next_token == sp.eos_id():
                    break
                generated_ids.append(next_token)
                curr_input = torch.cat([curr_input, torch.tensor([[next_token]], device=device)], dim=1)

            pred_word = sp.DecodeIds(generated_ids).strip()
            total += 1
            by_type[err_type]["total"] += 1

            if pred_word == target:
                correct += 1
                by_type[err_type]["correct"] += 1

    overall_acc = correct / max(1, total)
    return {
        "total_tested": total,
        "overall_accuracy": overall_acc,
        "breakdown": {k: v["correct"] / max(1, v["total"]) for k, v in by_type.items()}
    }


def interactive_repl(model: LlamaForCausalLM, sp: spm.SentencePieceProcessor, device: torch.device):
    """Interactive CLI REPL for live testing predictions and autocorrect."""
    bos_id = sp.bos_id() if sp.bos_id() != -1 else 1
    print("\n" + "=" * 60)
    print("Greek Keyboard LM - Interactive Console")
    print("Commands:")
    print("  - Type Greek text to see Top-5 next-word predictions")
    print("  - Type '<XBU>word<XBC>' to test autocorrect")
    print("  - Type 'exit' or 'quit' to end")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n[Prompt] > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break

            token_ids = [bos_id] + sp.EncodeAsIds(user_input)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_tensor).logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=5)

            print("Top 5 Predictions:")
            for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices), start=1):
                piece = sp.IdToPiece(idx.item())
                # Replace SentencePiece underscore with visible space representation
                readable_piece = piece.replace(" ", " ")
                print(f"  {rank}. '{readable_piece}' (ID: {idx.item()}, Prob: {prob.item()*100:.2f}%)")

        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate Greek Keyboard LM")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints/best_model",
                        help="Path to trained PyTorch model directory")
    parser.add_argument("--tokenizer_file", type=str, default="models/tokenizer/tokenizer.model",
                        help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--test_file", type=str, default="data/processed/test.txt",
                        help="Path to test split text file")
    parser.add_argument("--eval_jsonl", type=str, default="data/autocorrect_eval.jsonl",
                        help="Path to synthetic autocorrect evaluation benchmark")
    parser.add_argument("--max_test_samples", type=int, default=500,
                        help="Max test sentences for perplexity evaluation")
    parser.add_argument("--num_accent_samples", type=int, default=1000,
                        help="Number of samples for accent restoration benchmark")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI demo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_dir)
    tokenizer_path = Path(args.tokenizer_file)

    if not model_path.exists() or not tokenizer_path.exists():
        print(f"[!] Error: Model or tokenizer not found ({model_path}, {tokenizer_path})")
        print("[!] Please run 03_train_sentencepiece.py and 05_train_model.py first.")
        sys.exit(1)

    model, sp = load_model_and_tokenizer(model_path, tokenizer_path, device)

    if args.interactive:
        interactive_repl(model, sp, device)
        return

    print("\n" + "=" * 60)
    print("GREEK KEYBOARD LM EVALUATION REPORT")
    print("=" * 60)

    # 1. Perplexity & Top-K
    ppl_results = evaluate_perplexity_and_topk(
        model, sp, Path(args.test_file), device, max_samples=args.max_test_samples
    )
    if ppl_results:
        print("\n--- 1. Next-Word Prediction Benchmark ---")
        print(f"- Total Test Tokens: {ppl_results['test_tokens']:,}")
        print(f"- Perplexity: {ppl_results['perplexity']:.2f}")
        print(f"- Cross-Entropy Loss: {ppl_results['cross_entropy_loss']:.4f}")
        print(f"- Top-1 Accuracy: {ppl_results['top1_acc']*100:.2f}%")
        print(f"- Top-3 Accuracy: {ppl_results['top3_acc']*100:.2f}%")
        print(f"- Top-5 Accuracy: {ppl_results['top5_acc']*100:.2f}%")
        print(f"- Top-10 Accuracy: {ppl_results['top10_acc']*100:.2f}%")

    # 2. Accent Restoration
    accent_results = evaluate_accent_restoration(
        model, sp, Path(args.test_file), device, num_samples=args.num_accent_samples
    )
    if accent_results:
        print("\n--- 2. Accent Restoration Benchmark ---")
        print(f"- Words Evaluated: {accent_results['samples_evaluated']:,}")
        print(f"- Accent Restoration Accuracy: {accent_results['accent_accuracy']*100:.2f}%")
        status = "PASSED (>=95%)" if accent_results['target_met'] else "FAILED (<95%)"
        print(f"- Target Status: {status}")

    # 3. Autocorrect Benchmark
    ac_results = evaluate_synthetic_autocorrect(
        model, sp, Path(args.eval_jsonl), device, max_samples=args.max_test_samples
    )
    if ac_results:
        print("\n--- 3. Synthetic Typo Autocorrect Benchmark ---")
        print(f"- Samples Tested: {ac_results['total_tested']:,}")
        print(f"- Overall Accuracy: {ac_results['overall_accuracy']*100:.2f}%")
        for err_type, acc in ac_results.get("breakdown", {}).items():
            print(f"  * {err_type}: {acc*100:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
