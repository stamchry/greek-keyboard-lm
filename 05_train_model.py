#!/usr/bin/env python3
"""
05_train_model.py - 22M LLaMA Causal Language Model Training Pipeline for Greek Keyboard LM

Trains a custom 22M parameter LlamaForCausalLM model with:
- Architecture: hidden_size=512, layers=10, heads=8, intermediate_size=1376, vocab_size=15008, ctx=256
- Dynamic batch mixing: 70% clean next-word text + 30% synthetic autocorrect sequences (<XBU>...<XBC>...<XEC>)
- Cosine LR schedule with linear warmup, AdamW, mixed precision (bf16/fp16/fp32)
- Saves Hugging Face compatible checkpoints ready for GGUF conversion.
"""

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

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


class GreekKeyboardDataset(Dataset):
    """
    Dataset that dynamically blends:
    - 70% clean next-token prediction
    - 30% synthetic autocorrect sequences ([Context] <XBU>corrupted<XBC>target<XEC>)
    """

    def __init__(
        self,
        text_file: Path,
        sp_model_path: Path,
        max_seq_len: int = 256,
        autocorrect_ratio: float = 0.30,
        is_train: bool = True
    ):
        self.max_seq_len = max_seq_len
        self.autocorrect_ratio = autocorrect_ratio
        self.is_train = is_train

        # Load SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(sp_model_path))

        self.bos_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 2
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 3

        # Initialize corrupter
        self.corrupter = GreekCorrupter(seed=42) if GreekCorrupter is not None else None

        # Load sentences
        self.lines: List[str] = []
        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.lines.append(line)

        print(f"[+] Loaded {len(self.lines):,} lines from {text_file.name} (train={is_train})")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = self.lines[idx]

        # 30% chance to generate an autocorrect prompt during training
        use_autocorrect = (
            self.is_train and
            self.corrupter is not None and
            random.random() < self.autocorrect_ratio
        )

        if use_autocorrect:
            sample = self.corrupter.create_autocorrect_sample(line)
            if sample and sample["full_sequence"]:
                text_to_encode = sample["full_sequence"]
            else:
                text_to_encode = line
        else:
            text_to_encode = line

        # Encode tokens
        token_ids = [self.bos_id] + self.sp.EncodeAsIds(text_to_encode) + [self.eos_id]

        # Truncate if exceeds max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        # Input and target labels (shifted by 1 for autoregressive LM)
        seq_len = len(token_ids)
        input_ids = token_ids.copy()
        labels = token_ids.copy()

        # Pad to max_seq_len
        padding_len = self.max_seq_len - seq_len
        if padding_len > 0:
            input_ids += [self.pad_id] * padding_len
            labels += [-100] * padding_len  # -100 ignored by CrossEntropyLoss

        attention_mask = [1] * seq_len + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def build_llama_mini_config(
    vocab_size: int = 15008,
    max_position_embeddings: int = 256,
    num_hidden_layers: int = 9,
    hidden_size: int = 512,
    intermediate_size: int = 1376,
    num_attention_heads: int = 8,
    tie_word_embeddings: bool = True
) -> LlamaConfig:
    """
    Constructs the official ~36M parameter LLaMA configuration matching FUTO specifications:
    - hidden_size = 512
    - num_hidden_layers = 9
    - num_attention_heads = 8 (head_dim = 64)
    - intermediate_size = 1376 (SwiGLU)
    - max_position_embeddings = 256
    - tie_word_embeddings = True
    - Total parameters = ~36.15M
    """
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=tie_word_embeddings
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def train(
    train_file: Path,
    val_file: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 500,
    eval_every_steps: int = 250,
    save_every_steps: int = 500,
    max_seq_len: int = 256,
    num_layers: int = 9,
    hidden_size: int = 512,
    intermediate_size: int = 1376,
    tie_embeddings: bool = True,
    num_workers: int = 2,
    seed: int = 42
):
    torch.manual_seed(seed)
    random.seed(seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_amp = True
        print(f"[+] Using CUDA device: {torch.cuda.get_device_name(0)} with mixed precision ({amp_dtype})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_dtype = torch.float32
        use_amp = False
        print("[+] Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        use_amp = False
        print(f"[+] Using CPU device (Threads: {torch.get_num_threads()})")

    sp_model_file = tokenizer_dir / "tokenizer.model"
    if not sp_model_file.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {sp_model_file}")

    # Datasets and Loaders
    train_dataset = GreekKeyboardDataset(
        text_file=train_file,
        sp_model_path=sp_model_file,
        max_seq_len=max_seq_len,
        autocorrect_ratio=0.30,
        is_train=True
    )
    val_dataset = GreekKeyboardDataset(
        text_file=val_file,
        sp_model_path=sp_model_file,
        max_seq_len=max_seq_len,
        autocorrect_ratio=0.30,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Initialize Model
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_file))
    vocab_size = sp.GetPieceSize()

    config = build_llama_mini_config(
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tie_word_embeddings=tie_embeddings
    )
    model = LlamaForCausalLM(config).to(device)

    total_params, trainable_params = count_parameters(model)
    print("=" * 60)
    print(f"Model Architecture: ~36M Mini-LLaMA for FUTO Greek Keyboard")
    print(f"- Total Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"- Trainable Parameters: {trainable_params:,}")
    print(f"- Layers: {config.num_hidden_layers}, Hidden Size: {config.hidden_size}, Heads: {config.num_attention_heads}")
    print(f"- Intermediate Size: {config.intermediate_size}, Vocab Size: {config.vocab_size}")
    print(f"- Tied Embeddings: {config.tie_word_embeddings}")
    print("=" * 60)

    # Optimizer & Scheduler
    no_decay = ["bias", "layer_norm", "layernorm", "rms_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    total_training_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path = output_path / "best_model"

    best_val_loss = float("inf")
    global_step = 0

    print("[-] Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Periodic Evaluation
                if global_step % eval_every_steps == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                    print(f"\n[Step {global_step}] Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"[*] New best validation loss! Saving to {best_model_path}")
                        model.save_pretrained(best_model_path)
                        config.save_pretrained(best_model_path)
                    model.train()

                # Periodic Checkpoint
                if global_step % save_every_steps == 0:
                    ckpt_path = output_path / f"checkpoint_step_{global_step}"
                    model.save_pretrained(ckpt_path)
                    config.save_pretrained(ckpt_path)

        print(f"[✓] Completed Epoch {epoch}/{epochs}. Average Epoch Loss: {epoch_loss/len(train_loader):.4f}")

    # Final Save
    final_path = output_path / "final_model"
    model.save_pretrained(final_path)
    config.save_pretrained(final_path)
    print(f"[✓] Training complete. Final model saved to {final_path}")
    print(f"[✓] Best model checkpoint: {best_model_path} (Best Val Loss: {best_val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train ~36M Greek Keyboard LLaMA Model for FUTO")
    parser.add_argument("--train_file", type=str, default="data/processed/train.txt",
                        help="Path to training data text file")
    parser.add_argument("--val_file", type=str, default="data/processed/val.txt",
                        help="Path to validation data text file")
    parser.add_argument("--tokenizer_dir", type=str, default="models/tokenizer",
                        help="Directory containing trained SentencePiece tokenizer")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints",
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--eval_every", type=int, default=250, help="Evaluation frequency in steps")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint frequency in steps")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max context window")
    parser.add_argument("--num_layers", type=int, default=9, help="Number of Transformer layers (FUTO standard: 9)")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden dimension d_model (FUTO standard: 512)")
    parser.add_argument("--intermediate_size", type=int, default=1376, help="SwiGLU intermediate dimension (default: 1376)")
    parser.add_argument("--untie_embeddings", action="store_true", help="Untie input embedding and lm_head weights")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        tokenizer_dir=Path(args.tokenizer_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_every_steps=args.eval_every,
        save_every_steps=args.save_every,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        tie_embeddings=not args.untie_embeddings,
        num_workers=args.num_workers,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
