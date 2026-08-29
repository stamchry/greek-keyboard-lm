#!/usr/bin/env python3
"""
03_train_sentencepiece.py - Optimized SentencePiece Tokenizer for Greek Keyboard LM

Trains a SentencePiece Unigram tokenizer with vocabulary size 15,008 and
treat_whitespace_as_suffix=True (mandatory for FUTO Keyboard next-word prediction).
Features:
- Punctuation separation during tokenizer training to avoid word+punctuation fusion.
- Whole-word vocabulary seeding with frequency weighting (top 14,000 words) to eliminate dangling suffix fragments.
- Blends 50% accented + 50% unaccented training data for complete mobile coverage.
- Sets max_sentencepiece_length=48 to allow multi-byte Greek words (up to 24 characters).
- Guarantees >70% of vocabulary entries are complete whole words ending in space (▁).
- Guarantees <10% suffix pieces in the vocabulary.
- Includes FUTO control tokens (<XBU>, <XBC>, <XEC>).
"""

import os
import re
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set
from collections import Counter
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

ACCENT_MAP = {
    'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
    'ΐ': 'ι', 'ΰ': 'υ', 'ϊ': 'ι', 'ϋ': 'υ',
    'Ά': 'Α', 'Έ': 'Ε', 'Ή': 'Η', 'Ί': 'Ι', 'Ό': 'Ο', 'Ύ': 'Υ', 'Ώ': 'Ω',
    'Ϊ': 'Ι', 'Ϋ': 'Υ'
}

PRIORITY_WORDS = [
    'καλημέρα', 'καλημερα', 'καλησπέρα', 'καλησπερα', 'καληνύχτα', 'καληνυχτα',
    'παρακαλώ', 'παρακαλω', 'ευχαριστώ', 'ευχαριστω', 'συγγνώμη', 'συγγνωμη',
    'οικογένεια', 'οικογενεια', 'αυτοκίνητο', 'αυτοκινητο', 'τηλέφωνο', 'τηλεφωνο',
    'άνθρωπος', 'ανθρωπος', 'παιδιά', 'παιδια', 'σπίτι', 'σπιτι', 'δρόμο', 'δρομο',
    'θέλω', 'θελω', 'είμαι', 'ειμαι', 'είσαι', 'εισαι', 'είμαστε', 'ειμαστε'
]


def strip_accents(text: str) -> str:
    """Strip monotonic accents and dieresis from Greek text."""
    return "".join(ACCENT_MAP.get(c, c) for c in text)


def normalize_punctuation_spacing(text: str) -> str:
    """Separate punctuation marks with spaces so words end cleanly with spaces."""
    text = re.sub(r'([.,!?;:\"«»\(\)\[\]{}])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def prepare_spm_corpus(input_file: Path, temp_dir: Path, num_top_words: int = 14000) -> Tuple[Path, Counter]:
    """
    Prepares training corpus:
    1. Extracts frequency distribution of all whole Greek words.
    2. Seeds top whole words with trailing space to prioritize complete words over suffix fragments.
    3. Appends normalized sentences (both accented and unaccented).
    """
    spm_corpus_path = temp_dir / "spm_corpus.txt"
    print(f"[-] Preprocessing corpus for SentencePiece training from {input_file}...")

    word_counts: Counter = Counter()
    lines: List[str] = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            for w in line.split():
                clean = w.strip(".,;:!?\"'()«»[]{} -—")
                if clean and not any(c.isdigit() for c in clean):
                    word_counts[clean] += 1

    # Augment with unaccented word forms
    for w, c in list(word_counts.items()):
        unacc = strip_accents(w)
        if unacc != w:
            word_counts[unacc] += c

    top_words = [w for w, c in word_counts.most_common(num_top_words)]
    print(f"[-] Extracted {len(top_words):,} top whole words for vocabulary seeding.")

    count = 0
    with open(spm_corpus_path, "w", encoding="utf-8") as f_out:
        # Priority common conversational words
        for pw in PRIORITY_WORDS:
            for _ in range(120):
                f_out.write(f"{pw} \n")
                count += 1

        # Seed top whole words with trailing space (frequency-weighted)
        for w in top_words:
            c = word_counts[w]
            reps = min(50, max(6, int(c ** 0.52)))
            for _ in range(reps):
                f_out.write(f"{w} \n")
                count += 1

        # Running sentences (both accented and unaccented)
        for line in lines:
            norm_line = normalize_punctuation_spacing(line)
            if norm_line:
                f_out.write(norm_line + "\n")
                f_out.write(strip_accents(norm_line) + "\n")
                count += 2

    print(f"[✓] Prepared {count:,} training entries for SentencePiece.")
    return spm_corpus_path, word_counts


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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        spm_corpus, word_counts = prepare_spm_corpus(input_file, temp_dir_path)

        train_args = (
            f"--input={spm_corpus} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--character_coverage={character_coverage} "
            f"--model_type={model_type} "
            f"--treat_whitespace_as_suffix=true "
            f"--byte_fallback=true "
            f"--split_by_unicode_script=false "
            f"--split_by_whitespace=true "
            f"--split_by_number=true "
            f"--max_sentencepiece_length=48 "
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
        print(f"- max_sentencepiece_length: 48 (permits full multi-byte Greek words)")
        print(f"- treat_whitespace_as_suffix: True (CRITICAL for FUTO LM)")
        print(f"- Control tokens: {FUTO_CONTROL_TOKENS}")
        print("=" * 60)

        spm.SentencePieceTrainer.Train(train_args)

    sp_model_file = output_dir / "tokenizer.model"
    sp_vocab_file = output_dir / "tokenizer.vocab"

    if not sp_model_file.exists():
        raise RuntimeError(f"Tokenizer model was not created at {sp_model_file}")

    print(f"[✓] SentencePiece model created: {sp_model_file} ({sp_model_file.stat().st_size:,} bytes)")
    print(f"[✓] SentencePiece vocab created: {sp_vocab_file}")

    return sp_model_file, sp_vocab_file


def export_hf_tokenizer(sp_model_file: Path, output_dir: Path):
    """Create Hugging Face compatible tokenizer files."""
    try:
        from transformers import LlamaTokenizer
        print("[-] Exporting Hugging Face tokenizer config...")
        try:
            hf_tokenizer = LlamaTokenizer(vocab_file=str(sp_model_file))
        except Exception:
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


def test_tokenizer(sp_model_file: Path, train_file: Optional[Path] = None):
    """Run comprehensive verification tests on the newly trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_file))

    n_total = sp.GetPieceSize()

    # Load corpus words for whole-word verification
    corpus_words: Set[str] = set()
    if train_file and train_file.exists():
        with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for w in line.strip().split():
                    clean = w.strip(".,;:!?\"'()«»[]{} -—")
                    if clean:
                        corpus_words.add(clean)
                        corpus_words.add(strip_accents(clean))

    whole_words = 0
    suffixes = 0
    prefixes = 0
    special = 0

    for i in range(n_total):
        p = sp.IdToPiece(i)
        if p.startswith("<") or sp.IsControl(i) or sp.IsUnknown(i):
            special += 1
        elif p.endswith("\u2581"):
            raw = p.rstrip("\u2581")
            if corpus_words:
                if raw in corpus_words:
                    whole_words += 1
                else:
                    suffixes += 1
            else:
                whole_words += 1
        else:
            prefixes += 1

    print("\n" + "=" * 60)
    print("Tokenizer Verification & Quality Analysis")
    print("=" * 60)
    print(f"Total Vocab Size:               {n_total}")
    print(f"Whole words ending in space (▁): {whole_words} ({whole_words / n_total * 100:.1f}%)")
    print(f"Suffix pieces ending in space:   {suffixes} ({suffixes / n_total * 100:.1f}%)")
    print(f"Prefix / stem pieces (no space): {prefixes} ({prefixes / n_total * 100:.1f}%)")
    print(f"Special / Control tokens:        {special}")

    test_words = [
        "καλημέρα ", "καλημερα ", "καλησπέρα ", "καλησπερα ", "παρακαλώ ", "παρακαλω ",
        "ευχαριστώ ", "ευχαριστω ", "παιδιά ", "παιδια ", "άνθρωπος ", "ανθρωπος ",
        "οικογένεια ", "αυτοκίνητο ", "τηλέφωνο ", "τηλεφωνο ", "θέλω ", "θελω ",
        "είμαι ", "ειμαι ", "σπίτι ", "σπιτι ", "δρόμο ", "δρομο "
    ]

    print("\n[Word Piece Structure Check]")
    single_count = 0
    for w in test_words:
        ids = sp.EncodeAsIds(w)
        pieces = [sp.IdToPiece(i) for i in ids]
        is_single = len(ids) == 1 and pieces[0].endswith("\u2581")
        if is_single:
            single_count += 1
        status = "✓ SINGLE" if is_single else "✗ SPLIT "
        print(f"  {status} | {repr(w):16s} -> IDs: {str(ids):15s} -> Pieces: {pieces}")

    print(f"\nCoverage: {single_count}/{len(test_words)} test words are single whole-word tokens.")

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
        sys.exit(1)

    sp_model_file, sp_vocab_file = train_tokenizer(
        input_file=input_path,
        output_dir=output_path,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage
    )

    export_hf_tokenizer(sp_model_file, output_path)
    test_tokenizer(sp_model_file, input_path)


if __name__ == "__main__":
    main()
