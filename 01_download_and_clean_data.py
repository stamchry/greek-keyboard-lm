#!/usr/bin/env python3
"""
01_download_and_clean_data.py - Greek Dataset Ingestion and Quality Assurance Pipeline

Ingests conversational (OpenSubtitles, Tatoeba) and informational (Greek Wikipedia, mC4)
datasets, applies Greek character ratio filtering, subtitle artifact removal,
HTML/URL cleaning, length filtering, and hash-based deduplication.
Outputs cleaned train, validation, and test splits.
"""

import os
import re
import sys
import html
import json
import random
import hashlib
import argparse
import unicodedata
from pathlib import Path
from typing import Iterator, Set, Tuple, List, Optional
from tqdm import tqdm


def load_dotenv_if_present():
    """Automatically loads variables from .env file into os.environ if present."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and v and k not in os.environ:
                        os.environ[k] = v

load_dotenv_if_present()


# --- Regex Filters & Cleaning Rules ---

# Subtitle timestamp cues: 00:01:23,456 --> 00:01:25,789
RE_TIMESTAMPS = re.compile(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3}')
RE_STANDALONE_DIGITS = re.compile(r'^\d+$')

# Hearing-impaired tags & sound effects: [ΜΟΥΣΙΚΗ], (ΓΕΛΙΑ), {ΗΧΟΙ}, ♪...♪
RE_BRACKETS = re.compile(r'\[.*?\]|\(.*?\)|<.*?>|\{.*?\}|♪.*?♪|♫.*?♫')

# Speaker tags at the start of a line: "- ΝΙΚΟΣ:", "ΜΑΡΙΑ:", ">> ", "- "
RE_SPEAKER_PREFIX = re.compile(r'^\s*(?:[-–—»>]+\s*)?(?:[Α-ΩΆΈΉΊΌΎΏΪΫA-Z0-9_\s]{2,20}:|[-–—»>]+)\s*')

# Subtitle boilerplate / metadata headers
RE_SUBTITLE_CREDITS = re.compile(
    r'(?i)(?:υπότιτλοι|μετάφραση|συγχρονισμός|επιμέλεια|διορθώσεις|απόδοση|σχολιασμός|'
    r'subtitles|sync|ripped|encoded|downloaded|opensubtitles|addic7ed|podnapisi|tvsubtitles)'
)

# Web artifacts: URLs, emails, HTML entities
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Multiple whitespaces
RE_WHITESPACE = re.compile(r'\s+')


# Administrative dumps, electoral tables, legal gazettes, web boilerplate
RE_ADMIN_JUNK = re.compile(
    r'(?i)(?:εκλογικό τμήμα|εκλογικής περιφέρειας|δημοτικής ενότητας|δικηγορικός σύλλογος|'
    r'αρ\.?\s*πρωτ|α\.δ\.τ|α\.φ\.μ|φ\.ε\.κ|διαύγεια|αδα:|αριθμ\.\s*πραξ|'
    r'ημερομηνία δημοσίευσης|τελευταία ενημέρωση|συντάχθηκε:|όροι χρήσης|'
    r'πνευματικά δικαιώματα|cookies?|all rights reserved|τηλ:\s*\d|φαξ:\s*\d|'
    r'τ\.κ\.\s*\d|ονοματεπώνυμο|επιτυχόντες|αποφασίζουμε|διοικητικό συμβούλιο|'
    r'πρωτοδικεί|ειρηνοδικεί|συμβολαιογράφ|δελτίο τύπου|διάμετρος/διαδρομή)'
)

# Repeated characters (e.g. "αααα", "!!!!!")
RE_REPEATED_CHARS = re.compile(r'(.)\1{3,}')


def is_greek_char(char: str) -> bool:
    """Check if character is in Greek Unicode ranges."""
    code = ord(char)
    # Greek and Coptic (0370-03FF) or Greek Extended (1F00-1FFF)
    return (0x0370 <= code <= 0x03FF) or (0x1F00 <= code <= 0x1FFF)


def calculate_greek_ratio(text: str) -> float:
    """Calculate ratio of Greek alphabetic characters to all alphabetic characters."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    greek_chars = [c for c in alpha_chars if is_greek_char(c)]
    return len(greek_chars) / len(alpha_chars)


def clean_text_line(line: str) -> Optional[str]:
    """
    Clean and normalize a single text line.
    Returns cleaned line, or None if the line should be discarded.
    """
    if not line:
        return None

    # Unescape HTML entities
    line = html.unescape(line)

    # Strip URLs and emails
    line = RE_URL.sub('', line)
    line = RE_EMAIL.sub('', line)

    # Strip subtitle timestamps and standalone sequence numbers
    line = RE_TIMESTAMPS.sub('', line)
    if RE_STANDALONE_DIGITS.match(line.strip()):
        return None

    # Check for subtitle credits / boilerplate
    if RE_SUBTITLE_CREDITS.search(line):
        return None

    # Check for administrative dumps, gazettes, electoral tables
    if RE_ADMIN_JUNK.search(line):
        return None

    # Strip hearing-impaired tags and sound notations
    line = RE_BRACKETS.sub('', line)

    # Strip speaker prefixes
    line = RE_SPEAKER_PREFIX.sub('', line)

    # Normalize unicode (NFC)
    line = unicodedata.normalize('NFC', line)

    # Normalize Greek punctuation (standardize on Greek question mark ;)
    line = line.replace(';', ';').replace(';', ';')
    line = line.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"')
    line = line.replace('’', "'").replace('`', "'")

    # Collapse multiple whitespaces
    line = RE_WHITESPACE.sub(' ', line).strip()

    # Discard if too short
    if not line:
        return None

    # Discard lines with character repetitions (e.g. "αααααα")
    if RE_REPEATED_CHARS.search(line):
        return None

    # Length constraints: between 3 and 40 words
    words = line.split()
    if len(words) < 3 or len(words) > 40:
        return None

    # Greek character ratio constraint (>= 85% of letters must be Greek)
    if calculate_greek_ratio(line) < 0.85:
        return None

    # Uppercase ratio constraint: discard all-caps shouting and tabular records
    letters = [c for c in line if c.isalpha()]
    if letters:
        upper_letters = [c for c in letters if c.isupper()]
        if (len(upper_letters) / len(letters)) > 0.25 and len(words) > 3:
            return None

    # Digit / number density constraint (discard tables, phone numbers, ID lists)
    digit_count = sum(1 for c in line if c.isdigit())
    if (digit_count / len(line)) > 0.08:
        return None

    standalone_nums = [w for w in words if w.isdigit()]
    if len(standalone_nums) > 1:
        return None

    return line


def process_text_stream(lines: Iterator[str],
                        seen_hashes: Set[str],
                        max_samples: Optional[int] = None,
                        desc: str = "Processing") -> List[str]:
    """Clean and deduplicate a stream of raw text lines with progress tracking."""
    cleaned_lines: List[str] = []
    pbar = tqdm(total=max_samples, desc=desc, unit=" lines")
    
    for raw_line in lines:
        cleaned = clean_text_line(raw_line)
        if cleaned:
            # Exact deduplication via MD5 hash
            line_hash = hashlib.md5(cleaned.encode('utf-8')).hexdigest()
            if line_hash not in seen_hashes:
                seen_hashes.add(line_hash)
                cleaned_lines.append(cleaned)
                pbar.update(1)
                if max_samples and len(cleaned_lines) >= max_samples:
                    break
    pbar.close()
    return cleaned_lines


def load_hf_wikipedia(max_samples: int) -> Iterator[str]:
    """Stream articles from Greek Wikipedia."""
    try:
        from datasets import load_dataset
        print("[-] Fetching Greek Wikipedia...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.el", split="train", streaming=True)
        except Exception:
            ds = load_dataset("wikipedia", "20220301.el", split="train", streaming=True, trust_remote_code=True)

        for item in ds:
            text = item.get("text", "")
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    yield line
    except Exception as e:
        print(f"[!] Warning: Could not load Wikipedia from HF ({e}).")


def load_hf_conversational(max_samples: int) -> Iterator[str]:
    """Stream dialogues from Opus-100 Greek (subtitles, conversation, movies)."""
    from datasets import load_dataset
    print("[-] Fetching Conversational Dialogue dataset (Helsinki-NLP/opus-100)...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "el-en", split="train", streaming=True)
        for item in ds:
            translation = item.get("translation", {})
            el_text = translation.get("el", "")
            if el_text:
                yield el_text
    except Exception as e:
        print(f"[!] Warning: Could not load Opus-100 ({e})")


def load_hf_literature(max_samples: int) -> Iterator[str]:
    """Stream literary prose from Opus Books Greek."""
    from datasets import load_dataset
    print("[-] Fetching Literature & Prose dataset (Helsinki-NLP/opus_books)...")
    try:
        ds = load_dataset("Helsinki-NLP/opus_books", "el-en", split="train", streaming=True)
        for item in ds:
            translation = item.get("translation", {})
            el_text = translation.get("el", "")
            if el_text:
                yield el_text
    except Exception as e:
        print(f"[!] Warning: Could not load Opus Books ({e})")


def load_local_raw_files(raw_dir: Path) -> Iterator[str]:
    """Load text files from a local raw directory."""
    if not raw_dir.exists():
        return
    for ext in ("*.txt", "*.jsonl"):
        for file_path in raw_dir.glob(ext):
            print(f"[-] Ingesting local file: {file_path.name}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                text = data.get("text", "")
                                for subline in text.split("\n"):
                                    if subline.strip():
                                        yield subline.strip()
                            except Exception:
                                pass
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line


def generate_fallback_sample_data(count: int = 5000) -> List[str]:
    """Generate a high-quality starter sample dataset if online sources are unreachable."""
    base_dialogues = [
        "Καλημέρα, τι κάνεις σήμερα;",
        "Είμαι πολύ καλά, ευχαριστώ πολύ για το ενδιαφέρον.",
        "Πότε θα πάμε μια βόλτα στην παραλία;",
        "Νομίζω ότι το απόγευμα είναι η καλύτερη ώρα.",
        "Μην ξεχάσεις να πάρεις μαζί σου τα κλειδιά του σπιτιού.",
        "Έχω πολλή δουλειά στο γραφείο και δεν ξέρω αν θα προλάβω.",
        "Θέλεις να παραγγείλουμε φαγητό από το αγαπημένο μας μαγαζί;",
        "Η ελληνική γλώσσα έχει πλούσιο λεξιλόγιο και βαθιά ιστορία.",
        "Το πληκτρολόγιο για κινητές συσκευές πρέπει να είναι γρήγορο και ακριβές.",
        "Η αυτόματη διόρθωση βοηθάει στην άμεση διόρθωση των τονισμών και των λαθών.",
        "Πρέπει να διαβάσω το βιβλίο πριν ξεκινήσει το μάθημα.",
        "Ο καιρός σήμερα στην Αθήνα και στη Θεσσαλονίκη είναι εξαιρετικός.",
        "Στείλε μου ένα μήνυμα όταν φτάσεις στον προορισμό σου.",
        "Δεν υπάρχει κανένας λόγος να ανησυχείς για αυτό το θέμα.",
        "Όλα θα πάνε καλά και θα βρούμε την κατάλληλη λύση.",
        "Πού βρίσκεται ο πλησιέστερος σταθμός του μετρό;",
        "Αύριο το πρωί έχουμε μια πολύ σημαντική συνάντηση εργασίας.",
        "Μπορείς να μου εξηγήσεις πώς λειτουργεί αυτός ο αλγόριθμος;",
        "Η τεχνητή νοημοσύνη αλλάζει τον τρόπο με τον οποίο γράφουμε και επικοινωνούμε.",
        "Χρειάζομαι ένα φλιτζάνι ζεστό καφέ για να ξεκινήσω τη μέρα μου."
    ]
    extended = []
    for i in range(count):
        sentence = random.choice(base_dialogues)
        extended.append(sentence)
    return extended


def main():
    parser = argparse.ArgumentParser(description="Greek Dataset Ingestion & QA Cleaner")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save train.txt, val.txt, test.txt")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory containing optional local raw text files")
    parser.add_argument("--max_conv", type=int, default=200000,
                        help="Target sample count for Conversational Dialogue (Opus-100)")
    parser.add_argument("--max_lit", type=int, default=30000,
                        help="Target sample count for Literature & Books (Opus Books)")
    parser.add_argument("--max_wiki", type=int, default=70000,
                        help="Target sample count for Greek Wikipedia")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip online datasets and process only local raw_dir")
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    all_cleaned_data: List[str] = []

    print("=" * 60)
    print("Greek Keyboard LM - Curated Dataset Ingestion Pipeline")
    print("Blend: Conversational (70%) + Literature (10%) + Wikipedia (20%)")
    print("=" * 60)

    # 1. Local files
    local_stream = load_local_raw_files(raw_path)
    local_cleaned = process_text_stream(local_stream, seen_hashes, desc="Local Files")
    if local_cleaned:
        print(f"[+] Loaded {len(local_cleaned):,} lines from local raw directory.")
        all_cleaned_data.extend(local_cleaned)

    # 2. Online datasets if not skipped
    if not args.skip_download:
        # A. Conversational (~70%)
        print("\n--- 1/3 Ingesting Conversational Dialogue (Opus-100) ---")
        conv_cleaned = process_text_stream(
            load_hf_conversational(args.max_conv),
            seen_hashes,
            max_samples=args.max_conv,
            desc="Conversational"
        )
        print(f"[+] Cleaned Conversational Lines: {len(conv_cleaned):,}")
        all_cleaned_data.extend(conv_cleaned)

        # B. Literature (~10%)
        print("\n--- 2/3 Ingesting Literature & Prose (Opus Books) ---")
        lit_cleaned = process_text_stream(
            load_hf_literature(args.max_lit),
            seen_hashes,
            max_samples=args.max_lit,
            desc="Literature"
        )
        print(f"[+] Cleaned Literature Lines: {len(lit_cleaned):,}")
        all_cleaned_data.extend(lit_cleaned)

        # C. Wikipedia (~20%)
        print("\n--- 3/3 Ingesting Informational Articles (Greek Wikipedia) ---")
        wiki_cleaned = process_text_stream(
            load_hf_wikipedia(args.max_wiki),
            seen_hashes,
            max_samples=args.max_wiki,
            desc="Wikipedia"
        )
        print(f"[+] Cleaned Wikipedia Lines: {len(wiki_cleaned):,}")
        all_cleaned_data.extend(wiki_cleaned)

    # Fallback if no data collected (e.g., completely offline)
    if len(all_cleaned_data) == 0:
        print("[!] No data ingested from online or local files. Generating fallback sample corpus...")
        fallback_data = generate_fallback_sample_data(count=10000)
        all_cleaned_data.extend(fallback_data)

    print(f"\n[+] Total unique cleaned lines after QA filter: {len(all_cleaned_data):,}")

    # Shuffle before splitting
    random.shuffle(all_cleaned_data)

    # Train / Val / Test split: 95% Train, 2.5% Val, 2.5% Test
    total = len(all_cleaned_data)
    n_val = max(100, int(total * 0.025))
    n_test = max(100, int(total * 0.025))
    n_train = total - n_val - n_test

    train_data = all_cleaned_data[:n_train]
    val_data = all_cleaned_data[n_train:n_train + n_val]
    test_data = all_cleaned_data[n_train + n_val:]

    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"
    test_file = output_path / "test.txt"
    meta_file = output_path / "dataset_stats.json"

    print(f"[-] Writing {len(train_data):,} lines to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")

    print(f"[-] Writing {len(val_data):,} lines to {val_file}")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")

    print(f"[-] Writing {len(test_data):,} lines to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_data) + "\n")

    stats = {
        "total_lines": total,
        "train_lines": len(train_data),
        "val_lines": len(val_data),
        "test_lines": len(test_data),
        "min_words_per_line": 3,
        "max_words_per_line": 40,
        "min_greek_ratio": 0.85
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[✓] Data preparation complete. Statistics saved to {meta_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
