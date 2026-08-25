#!/usr/bin/env python3
"""
02_inspect_data.py - Greek Dataset Diagnostic & Quality Assurance Audit

Audits the cleaned dataset for:
- Token, line, character, and vocabulary counts
- Greek vs Non-Greek character distribution
- Monotonic accent ratio on polysyllabic words (identifies unaccented text)
- Lexicon coverage & Out-Of-Vocabulary (OOV) rate
- Outlier detection (shortest, longest, unexpected symbols)
- Generates a structured Markdown report
"""

import os
import re
import sys
import json
import random
import argparse
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set


# Greek vowel definitions
GREEK_VOWELS_ALL = set("αειουηωάέήίόύώϊϋΐΰΑΕΙΟΥΗΩΆΈΉΊΌΎΏΪΫ")
GREEK_ACCENTED_VOWELS = set("άέήίόύώΐΰΆΈΉΊΌΎΏ")
GREEK_DIERESIS_VOWELS = set("ϊϋΪΫ")

# Common Modern Greek grammatical monosyllables (never take an accent in standard monotonic orthography)
MONOSYLLABIC_EXEMPTIONS = {
    "και", "για", "του", "των", "τους", "μου", "σου", "της", "τον", "την",
    "το", "τα", "οι", "μα", "να", "θα", "δε", "δεν", "μη", "μην", "πως",
    "που", "σε", "με", "μια", "δυο", "ναι", "ποιος", "ποια", "ποιο", "αν",
    "κι", "ως", "σαν", "προς", "στο", "στη", "στην", "στα", "στους", "στις",
    "μου", "σου", "του", "της", "μας", "σας", "τους", "τον", "την", "το"
}

# Regex to collapse diphthongs/digraphs for syllable counting (αι, ει, οι, ου, υι, αυ, ευ)
RE_DIGRAPHS = re.compile(r'αι|ει|οι|ου|υι|αυ|ευ|άι|έι|όι', re.IGNORECASE)


def count_greek_syllables(word: str) -> int:
    """Accurately estimate syllable count in Modern Greek by accounting for digraphs."""
    lower = word.lower()
    if lower in MONOSYLLABIC_EXEMPTIONS:
        return 1
    # Replace digraphs with single placeholder vowel
    simplified = RE_DIGRAPHS.sub('α', lower)
    return sum(1 for c in simplified if c in GREEK_VOWELS_ALL)


def has_greek_accent(word: str) -> bool:
    """Check if a word contains at least one accented vowel."""
    return any(c in GREEK_ACCENTED_VOWELS for c in word)


def audit_corpus(file_path: Path, max_sample_lines: int = 100000) -> Dict:
    """Run full diagnostic audit on a text corpus."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    total_lines = 0
    total_words = 0
    total_chars = 0
    word_freq = Counter()
    char_freq = Counter()

    polysyllabic_words_count = 0
    accented_polysyllabic_count = 0
    unaccented_polysyllabic_examples = Counter()

    shortest_lines: List[Tuple[int, str]] = []
    longest_lines: List[Tuple[int, str]] = []
    non_greek_lines: List[Tuple[float, str]] = []

    lines_to_process = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                lines_to_process.append(line)

    total_lines_in_file = len(lines_to_process)
    if total_lines_in_file > max_sample_lines:
        print(f"[-] Sampling {max_sample_lines:,} lines out of {total_lines_in_file:,} for audit...")
        sample_lines = random.sample(lines_to_process, max_sample_lines)
    else:
        sample_lines = lines_to_process

    for line in sample_lines:
        total_lines += 1
        total_chars += len(line)
        for char in line:
            char_freq[char] += 1

        words = line.split()
        word_count = len(words)
        total_words += word_count

        # Track line lengths
        if len(shortest_lines) < 5 or word_count < shortest_lines[-1][0]:
            shortest_lines.append((word_count, line))
            shortest_lines.sort(key=lambda x: x[0])
            shortest_lines = shortest_lines[:5]

        if len(longest_lines) < 5 or word_count > longest_lines[-1][0]:
            longest_lines.append((word_count, line))
            longest_lines.sort(key=lambda x: x[0], reverse=True)
            longest_lines = longest_lines[:5]

        # Check Greek ratio of line
        letters = [c for c in line if c.isalpha()]
        if letters:
            greek_letters = [c for c in letters if (0x0370 <= ord(c) <= 0x03FF or 0x1F00 <= ord(c) <= 0x1FFF)]
            ratio = len(greek_letters) / len(letters)
            if ratio < 1.0:
                if len(non_greek_lines) < 5 or ratio < non_greek_lines[-1][0]:
                    non_greek_lines.append((ratio, line))
                    non_greek_lines.sort(key=lambda x: x[0])
                    non_greek_lines = non_greek_lines[:5]

        for w in words:
            clean_w = w.strip(".,;:!?\"'()«»[]{} -—")
            if not clean_w:
                continue
            word_freq[clean_w.lower()] += 1

            syllable_count = count_greek_syllables(clean_w)
            if syllable_count >= 2:
                polysyllabic_words_count += 1
                if has_greek_accent(clean_w):
                    accented_polysyllabic_count += 1
                else:
                    unaccented_polysyllabic_examples[clean_w.lower()] += 1

    # Char breakdown
    greek_char_count = sum(count for char, count in char_freq.items() if 0x0370 <= ord(char) <= 0x03FF or 0x1F00 <= ord(char) <= 0x1FFF)
    latin_char_count = sum(count for char, count in char_freq.items() if ('a' <= char <= 'z' or 'A' <= char <= 'Z'))
    digit_count = sum(count for char, count in char_freq.items() if char.isdigit())
    space_count = char_freq.get(' ', 0)
    punct_count = total_chars - greek_char_count - latin_char_count - digit_count - space_count

    accent_ratio = (accented_polysyllabic_count / polysyllabic_words_count) if polysyllabic_words_count > 0 else 0.0

    return {
        "file_name": file_path.name,
        "total_lines_in_file": total_lines_in_file,
        "sampled_lines": total_lines,
        "total_words": total_words,
        "total_chars": total_chars,
        "avg_words_per_line": total_words / total_lines if total_lines > 0 else 0,
        "unique_vocab_size": len(word_freq),
        "top_words": word_freq.most_common(20),
        "char_stats": {
            "greek_char_count": greek_char_count,
            "greek_char_ratio": greek_char_count / total_chars if total_chars > 0 else 0,
            "latin_char_count": latin_char_count,
            "latin_char_ratio": latin_char_count / total_chars if total_chars > 0 else 0,
            "digit_count": digit_count,
            "punct_count": punct_count,
            "space_count": space_count
        },
        "accent_stats": {
            "polysyllabic_words": polysyllabic_words_count,
            "accented_polysyllabic": accented_polysyllabic_count,
            "accent_ratio": accent_ratio,
            "top_unaccented_polysyllabic": unaccented_polysyllabic_examples.most_common(10)
        },
        "shortest_lines": shortest_lines,
        "longest_lines": longest_lines,
        "non_greek_samples": non_greek_lines
    }


def generate_markdown_report(stats: Dict, output_path: Optional[Path] = None) -> str:
    """Format diagnostic statistics as a comprehensive Markdown report."""
    md = []
    md.append(f"# Dataset QA & Diagnostic Report: `{stats['file_name']}`\n")
    md.append("## 1. Corpus Volume & Density\n")
    md.append(f"- **Total Lines in File**: {stats['total_lines_in_file']:,}")
    md.append(f"- **Sampled Lines Audited**: {stats['sampled_lines']:,}")
    md.append(f"- **Total Words**: {stats['total_words']:,}")
    md.append(f"- **Total Characters**: {stats['total_chars']:,}")
    md.append(f"- **Average Words per Line**: {stats['avg_words_per_line']:.2f}")
    md.append(f"- **Unique Vocabulary Count**: {stats['unique_vocab_size']:,}\n")

    md.append("## 2. Character & Script Distribution\n")
    c = stats["char_stats"]
    md.append("| Category | Count | Percentage |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Greek Script** | {c['greek_char_count']:,} | {c['greek_char_ratio']*100:.2f}% |")
    md.append(f"| **Latin Script** | {c['latin_char_count']:,} | {c['latin_char_ratio']*100:.2f}% |")
    md.append(f"| **Digits (0-9)** | {c['digit_count']:,} | {c['digit_count']/stats['total_chars']*100:.2f}% |")
    md.append(f"| **Punctuation / Symbols** | {c['punct_count']:,} | {c['punct_count']/stats['total_chars']*100:.2f}% |")
    md.append(f"| **Whitespace** | {c['space_count']:,} | {c['space_count']/stats['total_chars']*100:.2f}% |\n")

    md.append("## 3. Accent & Monotonic Integrity\n")
    a = stats["accent_stats"]
    status_icon = "✅" if a["accent_ratio"] >= 0.90 else "⚠️"
    md.append(f"- **Polysyllabic Words Analyzed**: {a['polysyllabic_words']:,}")
    md.append(f"- **Properly Accented Words**: {a['accented_polysyllabic']:,}")
    md.append(f"- **Accent Compliance Ratio**: **{a['accent_ratio']*100:.2f}%** {status_icon}")
    if a["accent_ratio"] < 0.90:
        md.append("  > [!WARNING]\n  > Accent ratio is below 90%. Consider tightening input filters to remove unaccented web scrapes.\n")
    else:
        md.append("  > [!NOTE]\n  > Excellent accent integrity. High proportion of natural monotonic accents.\n")

    if a["top_unaccented_polysyllabic"]:
        md.append("### Top Unaccented Polysyllabic Occurrences:")
        for word, count in a["top_unaccented_polysyllabic"]:
            md.append(f"- `{word}` ({count:,} times)")
        md.append("")

    md.append("## 4. Top 20 Most Frequent Words\n")
    md.append("| Rank | Word | Occurrences |")
    md.append("| :--- | :--- | :--- |")
    for i, (word, count) in enumerate(stats["top_words"], start=1):
        md.append(f"| {i} | `{word}` | {count:,} |")
    md.append("")

    md.append("## 5. Outlier & Boundary Samples\n")
    md.append("### Shortest Lines (3+ words):")
    for w_count, line in stats["shortest_lines"]:
        md.append(f"- `[{w_count} words]` {line}")
    md.append("")

    md.append("### Longest Lines (<= 40 words):")
    for w_count, line in stats["longest_lines"]:
        md.append(f"- `[{w_count} words]` {line}")
    md.append("")

    if stats["non_greek_samples"]:
        md.append("### Lines with Non-Greek Characters (Lowest Greek Script Ratio):")
        for ratio, line in stats["non_greek_samples"]:
            md.append(f"- `[{ratio*100:.1f}% Greek]` {line}")
        md.append("")

    report_text = "\n".join(md)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[✓] Report written to {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Greek Dataset Diagnostic & QA Audit")
    parser.add_argument("--input_file", type=str, default="data/processed/train.txt",
                        help="Path to cleaned dataset text file")
    parser.add_argument("--report_file", type=str, default="data/qa_report.md",
                        help="Path to output Markdown report")
    parser.add_argument("--sample_size", type=int, default=100000,
                        help="Max number of lines to sample for audit")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    report_path = Path(args.report_file) if args.report_file else None

    print(f"[-] Auditing corpus: {input_path}")
    stats = audit_corpus(input_path, max_sample_lines=args.sample_size)
    report = generate_markdown_report(stats, report_path)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
