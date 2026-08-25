#!/usr/bin/env python3
"""
04_generate_corruptions.py - Synthetic Greek Autocorrect & Accent Distortion Engine

Generates synthetic Greek typing errors, unaccented pairs, keyboard adjacency typos,
homophone/iotacism confusions, and character transpositions/deletions.
Formats samples as FUTO control sequences: [Context] <XBU>[corrupted_word]<XBC>[target_word]<XEC>
"""

import re
import json
import random
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Accent stripping mapping for Greek characters
ACCENT_MAP = {
    'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
    'ΐ': 'ι', 'ΰ': 'υ', 'ϊ': 'ι', 'ϋ': 'υ',
    'Ά': 'Α', 'Έ': 'Ε', 'Ή': 'Η', 'Ί': 'Ι', 'Ό': 'Ο', 'Ύ': 'Υ', 'Ώ': 'Ω',
    'Ϊ': 'Ι', 'Ϋ': 'Υ'
}

# Greek QWERTY Layout Adjacency Graph
# Key layout:
# Row 1: ; ς ε ρ τ υ θ ι ο π
# Row 2: α σ δ φ γ η ξ κ λ
# Row 3: ζ χ ψ ω β ν μ
GREEK_KEYBOARD_ADJACENCY = {
    'α': ['σ', 'ζ', 'ς', 'ε', 'δ'],
    'σ': ['α', 'δ', 'ε', 'ρ', 'χ', 'ζ'],
    'δ': ['σ', 'φ', 'ρ', 'τ', 'χ', 'ψ'],
    'φ': ['δ', 'γ', 'τ', 'υ', 'ψ', 'ω'],
    'γ': ['φ', 'η', 'υ', 'θ', 'ω', 'β'],
    'η': ['γ', 'ξ', 'θ', 'ι', 'β', 'ν'],
    'ξ': ['η', 'κ', 'ι', 'ο', 'ν', 'μ'],
    'κ': ['ξ', 'λ', 'ο', 'π', 'μ'],
    'λ': ['κ', 'π', 'ο'],
    'ς': ['α', 'ε', 'σ'],
    'ε': ['ς', 'ρ', 'α', 'σ', 'δ'],
    'ρ': ['ε', 'τ', 'σ', 'δ', 'φ'],
    'τ': ['ρ', 'υ', 'δ', 'φ', 'γ'],
    'υ': ['τ', 'θ', 'φ', 'γ', 'η'],
    'θ': ['υ', 'ι', 'γ', 'η', 'ξ'],
    'ι': ['θ', 'ο', 'η', 'ξ', 'κ'],
    'ο': ['ι', 'π', 'ξ', 'κ', 'λ'],
    'π': ['ο', 'κ', 'λ'],
    'ζ': ['α', 'σ', 'χ'],
    'χ': ['ζ', 'ψ', 'σ', 'δ'],
    'ψ': ['χ', 'ω', 'δ', 'φ'],
    'ω': ['ψ', 'β', 'φ', 'γ'],
    'β': ['ω', 'ν', 'γ', 'η'],
    'ν': ['β', 'μ', 'η', 'ξ'],
    'μ': ['ν', 'ξ', 'κ']
}

# Homophone / Iotacism substitution patterns (Modern Greek phonology)
HOMOPHONE_MAP = [
    # /i/ sound
    ('ει', 'ι'), ('ι', 'ει'),
    ('οι', 'ι'), ('ι', 'οι'),
    ('η', 'ι'), ('ι', 'η'),
    ('υ', 'ι'), ('ι', 'υ'),
    ('ει', 'η'), ('η', 'ει'),
    ('οι', 'η'), ('η', 'οι'),
    ('υ', 'η'), ('η', 'υ'),
    # /e/ sound
    ('αι', 'ε'), ('ε', 'αι'),
    # /o/ sound
    ('ο', 'ω'), ('ω', 'ο')
]


class GreekCorrupter:
    """Generates synthetic typing errors matching mobile typing patterns in Modern Greek."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def strip_accents(text: str) -> str:
        """Strip all Greek monotonic accents and dieresis."""
        result = []
        for c in text:
            result.append(ACCENT_MAP.get(c, c))
        return "".join(result)

    @staticmethod
    def has_accents(text: str) -> bool:
        """Check if string contains any Greek accented characters."""
        return any(c in ACCENT_MAP for c in text)

    def corrupt_accents(self, word: str) -> Optional[str]:
        """Strip accents (50% of corruptions)."""
        stripped = self.strip_accents(word)
        if stripped != word:
            return stripped
        return None

    def corrupt_adjacency(self, word: str) -> Optional[str]:
        """Substitute a random character with an adjacent Greek keyboard key."""
        if len(word) == 0:
            return None
        
        # Pick a random character index
        indices = list(range(len(word)))
        random.shuffle(indices)

        for idx in indices:
            c = word[idx]
            base_c = ACCENT_MAP.get(c, c).lower()
            if base_c in GREEK_KEYBOARD_ADJACENCY:
                replacement = random.choice(GREEK_KEYBOARD_ADJACENCY[base_c])
                if c.isupper():
                    replacement = replacement.upper()
                return word[:idx] + replacement + word[idx + 1:]
        return None

    def corrupt_homophones(self, word: str) -> Optional[str]:
        """Confuse phonetically equivalent Greek vowel/diphthong sounds (Iotacism)."""
        lower_word = word.lower()
        candidates = []

        for src, dst in HOMOPHONE_MAP:
            if src in lower_word:
                candidates.append((src, dst))

        if not candidates:
            return None

        src, dst = random.choice(candidates)
        # Replace first occurrence
        idx = lower_word.find(src)
        if idx != -1:
            # Preserve case of first character
            if word[idx].isupper():
                dst = dst.capitalize()
            corrupted = word[:idx] + dst + word[idx + len(src):]
            if corrupted != word:
                return corrupted
        return None

    def corrupt_dynamics(self, word: str) -> Optional[str]:
        """Simulate mechanical typing typos: transposition, deletion, insertion, repetition."""
        if len(word) < 2:
            return None

        typo_type = random.choice(["transposition", "deletion", "insertion", "repetition"])
        chars = list(word)

        if typo_type == "transposition" and len(chars) >= 2:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)

        elif typo_type == "deletion" and len(chars) >= 3:
            idx = random.randint(0, len(chars) - 1)
            del chars[idx]
            return "".join(chars)

        elif typo_type == "repetition":
            idx = random.randint(0, len(chars) - 1)
            chars.insert(idx, chars[idx])
            return "".join(chars)

        elif typo_type == "insertion":
            idx = random.randint(0, len(chars))
            base_c = ACCENT_MAP.get(chars[min(idx, len(chars) - 1)], 'α').lower()
            adjacent_keys = GREEK_KEYBOARD_ADJACENCY.get(base_c, ['α', 'σ', 'ε'])
            inserted_char = random.choice(adjacent_keys)
            chars.insert(idx, inserted_char)
            return "".join(chars)

        return None

    def corrupt_word(self, word: str) -> Tuple[Optional[str], str]:
        """
        Corrupt a single word according to plan distribution:
        - 50% Accent stripping
        - 30% Keyboard adjacency
        - 10% Homophone confusions
        - 10% Typo dynamics
        Returns: (corrupted_word, error_type)
        """
        clean_word = word.strip(".,;:!?\"'()«»[]{} -—")
        if len(clean_word) < 2:
            return None, "none"

        dice = random.random()

        # 50% Accent stripping (if word has accents, otherwise fallback to other types)
        if dice < 0.50 and self.has_accents(clean_word):
            c = self.corrupt_accents(clean_word)
            if c:
                return c, "accent_strip"

        # 30% Keyboard adjacency
        if dice < 0.80:
            c = self.corrupt_adjacency(clean_word)
            if c:
                return c, "adjacency"

        # 10% Homophones
        if dice < 0.90:
            c = self.corrupt_homophones(clean_word)
            if c:
                return c, "homophone"

        # 10% Typo dynamics
        c = self.corrupt_dynamics(clean_word)
        if c:
            return c, "dynamics"

        # Fallback if specific corruption failed
        if self.has_accents(clean_word):
            c = self.corrupt_accents(clean_word)
            if c:
                return c, "accent_strip"

        c = self.corrupt_adjacency(clean_word)
        if c:
            return c, "adjacency"

        return None, "none"

    def format_autocorrect_prompt(self, context: str, corrupted_word: str, target_word: str) -> str:
        """Format into standard FUTO autocorrect protocol string."""
        context_str = context.strip()
        if context_str:
            return f"{context_str} <XBU>{corrupted_word}<XBC>{target_word}<XEC>"
        else:
            return f"<XBU>{corrupted_word}<XBC>{target_word}<XEC>"

    def create_autocorrect_sample(self, sentence: str) -> Optional[Dict]:
        """
        Pick a target word from a full sentence and construct a FUTO autocorrect sample.
        """
        words = sentence.split()
        if len(words) < 2:
            return None

        # Pick a word index (prefer words with length >= 3)
        candidate_indices = [i for i, w in enumerate(words) if len(w.strip(".,;:!?\"'()«»[]{} -—")) >= 3]
        if not candidate_indices:
            candidate_indices = list(range(len(words)))

        target_idx = random.choice(candidate_indices)
        raw_word = words[target_idx]
        
        # Split punctuation from word
        match = re.match(r'^([^\w]*)([\w]+)([^\w]*)$', raw_word)
        if not match:
            clean_word = raw_word
            prefix, suffix = "", ""
        else:
            prefix, clean_word, suffix = match.groups()

        corrupted_word, error_type = self.corrupt_word(clean_word)
        if not corrupted_word or corrupted_word == clean_word:
            return None

        # Reconstruct context preceding target word
        context_words = words[:target_idx]
        context = " ".join(context_words)
        if prefix:
            context = f"{context} {prefix}".strip()

        target_with_suffix = f"{clean_word}{suffix}"
        corrupted_with_suffix = f"{corrupted_word}{suffix}"

        full_prompt = self.format_autocorrect_prompt(context, corrupted_with_suffix, target_with_suffix)

        return {
            "context": context,
            "corrupted_word": corrupted_with_suffix,
            "target_word": target_with_suffix,
            "error_type": error_type,
            "full_sequence": full_prompt
        }


def generate_evaluation_set(corpus_file: Path, output_file: Path, num_samples: int = 1000):
    """Generate a fixed evaluation benchmark with 1000 accent restorations and typos."""
    corrupter = GreekCorrupter(seed=1337)
    
    if not corpus_file.exists():
        print(f"[!] Warning: Corpus file {corpus_file} not found. Generating standalone samples.")
        sample_sentences = [
            "Καλημέρα σας, πώς είστε σήμερα;",
            "Θέλω να πάμε μια βόλτα στο κέντρο της πόλης.",
            "Η ελληνική γλώσσα είναι πολύ όμορφη και πλούσια.",
            "Δεν μπορώ να βρω τα κλειδιά του αυτοκινήτου μου.",
            "Το νέο πληκτρολόγιο έχει εξαιρετική αυτόματη διόρθωση."
        ] * (num_samples // 5 + 10)
    else:
        with open(corpus_file, "r", encoding="utf-8") as f:
            sample_sentences = [line.strip() for line in f if line.strip()]

    eval_samples = []
    seen = set()

    print(f"[-] Generating {num_samples} evaluation benchmark samples...")
    random.shuffle(sample_sentences)

    for sentence in sample_sentences:
        sample = corrupter.create_autocorrect_sample(sentence)
        if sample:
            key = (sample["context"], sample["corrupted_word"], sample["target_word"])
            if key not in seen:
                seen.add(key)
                eval_samples.append(sample)
                if len(eval_samples) >= num_samples:
                    break

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[✓] Saved {len(eval_samples)} evaluation samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Greek Typo and Autocorrect Generator")
    parser.add_argument("--input_file", type=str, default="data/processed/test.txt",
                        help="Input text file to extract evaluation samples from")
    parser.add_argument("--output_file", type=str, default="data/autocorrect_eval.jsonl",
                        help="Path to save evaluation dataset")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of evaluation samples to produce")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    generate_evaluation_set(Path(args.input_file), Path(args.output_file), args.num_samples)


if __name__ == "__main__":
    main()
