#!/usr/bin/env python3
"""
03_train_sentencepiece.py - SentencePiece Tokenizer Trainer for Greek Keyboard LM

Trains a SentencePiece Unigram tokenizer with vocabulary size 15,008 and
treat_whitespace_as_suffix=True (mandatory for FUTO Keyboard word completion).
Includes FUTO control tokens (<XBU>, <XBC>, <XEC>) and exports both SentencePiece binary
and Hugging Face Tokenizer artifacts.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import sentencepiece as spm


SPECIAL_TOKENS = {
    "unk_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "pad_id": 3,
    "unk_piece": "<unk>",
    "bos_piece": "<s>",
    "eos_piece": "</s>",
    "pad_piece": "<pad>"
}

FUTO_CONTROL_TOKENS = ["<XBU>", "<XBC>", "<XEC>"]


def train_tokenizer(
    input_file: Path,
    output_dir: Path,
    vocab_size: int = 15008,
    character_coverage: float = 0.9999,
    model_type: str = "unigram",
    max_sentence_length: int = 4192
) -> Tuple[Path, Path]:
    """Train SentencePiece tokenizer and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "tokenizer"

    user_defined_symbols = ",".join(FUTO_CONTROL_TOKENS)

    train_args = (
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        f"--model_type={model_type} "
        f"--treat_whitespace_as_suffix=true "
        f"--byte_fallback=true "
        f"--split_by_unicode_script=true "
        f"--split_by_whitespace=true "
        f"--split_by_number=true "
        f"--user_defined_symbols={user_defined_symbols} "
        f"--unk_id={SPECIAL_TOKENS['unk_id']} "
        f"--bos_id={SPECIAL_TOKENS['bos_id']} "
        f"--eos_id={SPECIAL_TOKENS['eos_id']} "
        f"--pad_id={SPECIAL_TOKENS['pad_id']} "
        f"--unk_piece={SPECIAL_TOKENS['unk_piece']} "
        f"--bos_piece={SPECIAL_TOKENS['bos_piece']} "
        f"--eos_piece={SPECIAL_TOKENS['eos_piece']} "
        f"--pad_piece={SPECIAL_TOKENS['pad_piece']} "
        f"--max_sentence_length={max_sentence_length} "
        f"--input_sentence_size=5000000 "
        f"--shuffle_input_sentence=true "
        f"--hard_vocab_limit=false"
    )

    print("=" * 60)
    print("Training SentencePiece Tokenizer for FUTO Keyboard LM")
    print(f"- Input corpus: {input_file}")
    print(f"- Output directory: {output_dir}")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Model type: {model_type}")
    print(f"- treat_whitespace_as_suffix: True (CRITICAL)")
    print(f"- Control tokens: {FUTO_CONTROL_TOKENS}")
    print("=" * 60)

    spm.SentencePieceTrainer.Train(train_args)

    sp_model_file = output_dir / "tokenizer.model"
    sp_vocab_file = output_dir / "tokenizer.vocab"

    # Verify generated files
    if not sp_model_file.exists():
        raise RuntimeError(f"Tokenizer model was not created at {sp_model_file}")

    print(f"[✓] SentencePiece model created: {sp_model_file} ({sp_model_file.stat().st_size:,} bytes)")
    print(f"[✓] SentencePiece vocab created: {sp_vocab_file}")

    return sp_model_file, sp_vocab_file


def export_hf_tokenizer(sp_model_file: Path, output_dir: Path):
    """Create Hugging Face compatible tokenizer files."""
    try:
        from transformers import LlamaTokenizerFast, LlamaTokenizer
        print("[-] Exporting Hugging Face tokenizer config...")
        try:
            hf_tokenizer = LlamaTokenizer(vocab_file=str(sp_model_file))
        except Exception:
            from transformers import PreTrainedTokenizerFast
            hf_tokenizer = None

        tokenizer_config = {
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": FUTO_CONTROL_TOKENS,
            "model_max_length": 256,
            "tokenizer_class": "LlamaTokenizer"
        }
        with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)

        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": FUTO_CONTROL_TOKENS
        }
        with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2)

        print("[✓] Hugging Face tokenizer configuration exported.")
    except Exception as e:
        print(f"[!] Warning: Could not export HF tokenizer config ({e})")


def test_tokenizer(sp_model_file: Path):
    """Run verification tests on the newly trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_file))

    print("\n--- Tokenizer Self-Test ---")
    print(f"Loaded vocab size: {sp.GetPieceSize()}")

    test_sentences = [
        "Καλημέρα, τι κάνεις σήμερα;",
        "Η αυτόματη διόρθωση στο FUTO keyboard είναι εκπληκτική.",
        "Είμαι στο δρόμο και έρχομαι.",
        "καλημερα <XBU>καλημερα<XBC>καλημέρα<XEC>"
    ]

    for s in test_sentences:
        pieces = sp.EncodeAsPieces(s)
        ids = sp.EncodeAsIds(s)
        decoded = sp.DecodePieces(pieces)
        print(f"\nOriginal: '{s}'")
        print(f"Pieces ({len(pieces)}): {pieces}")
        print(f"IDs: {ids}")
        print(f"Decoded: '{decoded}'")
        assert decoded.strip() == s.strip(), f"Roundtrip decoding failed: '{decoded}' != '{s}'"

    # Verify control token IDs
    for token in FUTO_CONTROL_TOKENS:
        token_id = sp.PieceToId(token)
        print(f"Control token '{token}' -> ID: {token_id}")
        assert token_id != sp.unk_id(), f"Control token {token} was mapped to UNK!"

    print("\n[✓] All tokenizer tests passed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizer for Greek Keyboard LM")
    parser.add_argument("--input_file", type=str, default="data/processed/train.txt",
                        help="Path to training text file")
    parser.add_argument("--output_dir", type=str, default="models/tokenizer",
                        help="Output directory for tokenizer artifacts")
    parser.add_argument("--vocab_size", type=int, default=15008,
                        help="Vocabulary size (default: 15008)")
    parser.add_argument("--character_coverage", type=float, default=0.9999,
                        help="Character coverage (default: 0.9999)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"[!] Error: Training data file not found at {input_path}")
        print("[!] Please run 01_download_and_clean_data.py first.")
        sys.exit(1)

    sp_model_file, sp_vocab_file = train_tokenizer(
        input_file=input_path,
        output_dir=output_path,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage
    )

    export_hf_tokenizer(sp_model_file, output_path)
    test_tokenizer(sp_model_file)


if __name__ == "__main__":
    from typing import Tuple
    main()
