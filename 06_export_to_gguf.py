#!/usr/bin/env python3
"""
06_export_to_gguf.py - Convert PyTorch LLaMA Checkpoint to GGUF with FUTO KeyboardLM Metadata

Converts trained LlamaForCausalLM weights to standard GGUF format and embeds:
- Complete LLaMA architecture hyperparameters
- SentencePiece vocabulary and special tokens
- FUTO KeyboardLM custom metadata:
    - general.name / general.description / general.author / general.license
    - keyboardlm.languages = "el"
    - keyboardlm.features = "base_v1 inverted_space xbu_char_autocorrect_v1 lora_finetunable_v1"
    - keyboardlm.ext_tokenizer_type = "sentencepiece"
    - keyboardlm.ext_tokenizer_data = <raw tokenizer.model binary>
    - keyboardlm.finetuning_count = 0
    - keyboardlm.history = ""
- Optional quantization step to Q6_K and Q8_0 via llama-quantize
"""

import os
import sys
import json
import shutil
import struct
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import gguf
from transformers import LlamaConfig, LlamaForCausalLM
import sentencepiece as spm


# PyTorch to GGUF tensor name mappings
TENSOR_NAME_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight"
}

LAYER_TENSOR_NAME_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight"
}


def map_tensor_name(pt_name: str) -> str:
    """Map PyTorch parameter name to standard GGUF / GGML tensor name."""
    if pt_name in TENSOR_NAME_MAP:
        return TENSOR_NAME_MAP[pt_name]

    if pt_name.startswith("model.layers."):
        parts = pt_name.split(".", 3)
        layer_idx = parts[2]
        rest = parts[3]
        if rest in LAYER_TENSOR_NAME_MAP:
            return f"blk.{layer_idx}.{LAYER_TENSOR_NAME_MAP[rest]}"

    return pt_name


def load_model_weights(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model state dict from safetensors or pytorch_model.bin."""
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        print(f"[-] Loading weights from {safetensors_path}")
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    elif bin_path.exists():
        print(f"[-] Loading weights from {bin_path}")
        return torch.load(str(bin_path), map_location="cpu")
    else:
        # Try loading via HuggingFace
        print(f"[-] Loading model via AutoModel from {model_dir}")
        model = LlamaForCausalLM.from_pretrained(str(model_dir))
        return model.state_dict()


def export_to_gguf(
    model_dir: Path,
    tokenizer_file: Path,
    output_file: Path,
    out_type: str = "f16",
    model_name: str = "Greek Keyboard LM",
    author: str = "Community",
    description: str = "Modern Greek Transformer LM for FUTO Keyboard",
    languages: str = "el",
    features: str = "base_v1 inverted_space lora_finetunable_v1"
):
    """Converts PyTorch checkpoint to GGUF format and writes FUTO metadata."""
    print("=" * 60)
    print(f"Exporting PyTorch Checkpoint to GGUF: {output_file}")
    print("=" * 60)

    # 1. Load config
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config = LlamaConfig(**config_data)

    # 2. Load Tokenizer & Binary bytes
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {tokenizer_file}")
    
    with open(tokenizer_file, "rb") as f:
        tokenizer_bytes = f.read()

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_file))
    vocab_size = sp.GetPieceSize()

    # 3. Load Model weights
    state_dict = load_model_weights(model_dir)

    # 4. Initialize GGUF Writer
    gguf_writer = gguf.GGUFWriter(output_file, "llama")

    # A. Architecture metadata
    gguf_writer.add_name(model_name)
    gguf_writer.add_description(description)
    gguf_writer.add_author(author)
    gguf_writer.add_license("Apache-2.0")

    gguf_writer.add_block_count(config.num_hidden_layers)
    gguf_writer.add_context_length(config.max_position_embeddings)
    gguf_writer.add_embedding_length(config.hidden_size)
    gguf_writer.add_feed_forward_length(config.intermediate_size)
    gguf_writer.add_head_count(config.num_attention_heads)
    gguf_writer.add_head_count_kv(getattr(config, "num_key_value_heads", config.num_attention_heads))
    gguf_writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    gguf_writer.add_rope_dimension_count(config.hidden_size // config.num_attention_heads)

    # B. FUTO Custom Metadata (Mandatory)
    print("[-] Injecting FUTO KeyboardLM metadata...")
    gguf_writer.add_string("keyboardlm.languages", languages)
    gguf_writer.add_string("keyboardlm.features", features)
    gguf_writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    # Store raw bytes as uint8 array for FUTO Keyboard SentencePiece loader
    gguf_writer.add_array("keyboardlm.ext_tokenizer_data", tokenizer_bytes)
    gguf_writer.add_uint32("keyboardlm.finetuning_count", 0)
    gguf_writer.add_string("keyboardlm.history", "")

    # C. Tokenizer Metadata for GGUF
    print("[-] Extracting SentencePiece vocabulary...")
    tokens: List[str] = []
    scores: List[float] = []
    toktypes: List[int] = []

    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        tokens.append(piece)
        scores.append(score)

        if sp.IsUnknown(i):
            toktypes.append(gguf.TokenType.UNKNOWN)
        elif sp.IsControl(i):
            toktypes.append(gguf.TokenType.CONTROL)
        elif sp.IsUnused(i):
            toktypes.append(gguf.TokenType.UNUSED)
        elif sp.IsByte(i):
            toktypes.append(gguf.TokenType.BYTE)
        elif piece in ["<XBU>", "<XBC>", "<XEC>"]:
            toktypes.append(gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

    gguf_writer.add_tokenizer_model("llama")
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
    gguf_writer.add_bos_token_id(sp.bos_id() if sp.bos_id() != -1 else 1)
    gguf_writer.add_eos_token_id(sp.eos_id() if sp.eos_id() != -1 else 2)
    gguf_writer.add_unk_token_id(sp.unk_id() if sp.unk_id() != -1 else 0)
    gguf_writer.add_pad_token_id(sp.pad_id() if sp.pad_id() != -1 else 3)

    # D. Model Tensors
    print(f"[-] Writing tensors with data type {out_type}...")
    dtype_map = {
        "f32": np.float32,
        "f16": np.float16
    }
    target_np_dtype = dtype_map.get(out_type, np.float16)

    tensor_names_added = set()
    for pt_name, tensor in state_dict.items():
        gguf_name = map_tensor_name(pt_name)
        tensor_names_added.add(gguf_name)
        data = tensor.detach().cpu().numpy()

        # In GGUF, 1D norm weights (RMSNorm) must always be float32 for GGML binary ops
        if "norm" in gguf_name:
            data = data.astype(np.float32)
        elif data.dtype in (np.float32, np.float64, np.float16):
            data = data.astype(target_np_dtype)

        gguf_writer.add_tensor(gguf_name, data)

    # CRITICAL FOR FUTO KEYBOARD: If output.weight (lm_head) was tied, explicitly duplicate it as output.weight
    if "output.weight" not in tensor_names_added:
        print("[-] Tied embeddings detected: explicitly duplicating token_embd.weight as output.weight for FUTO Keyboard runtime...")
        embed_tensor = state_dict["model.embed_tokens.weight"].detach().cpu().numpy()
        embed_data = embed_tensor.astype(target_np_dtype)
        gguf_writer.add_tensor("output.weight", embed_data)
        tensor_names_added.add("output.weight")

    print(f"[-] Total tensors written to GGUF: {len(tensor_names_added)}")

    # Write GGUF file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print(f"[✓] Successfully exported GGUF model: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")


def quantize_gguf(input_gguf: Path, output_dir: Path):
    """
    Quantize the exported GGUF model to Q6_K and Q8_0.
    Checks for llama-quantize binary.
    """
    quantize_bin = shutil.which("llama-quantize") or shutil.which("llama.cpp/llama-quantize")
    
    q_types = ["Q6_K", "Q8_0"]

    if not quantize_bin:
        print("\n" + "=" * 60)
        print("[!] Note: 'llama-quantize' was not found in PATH.")
        print("[!] To quantize your model to Q6_K and Q8_0, build llama.cpp and run:")
        for q_type in q_types:
            out_name = output_dir / f"{input_gguf.stem}_{q_type}.gguf"
            print(f"    llama-quantize {input_gguf} {out_name} {q_type}")
        print("=" * 60)
        return

    print("\n[-] Quantizing GGUF models...")
    for q_type in q_types:
        base_stem = input_gguf.stem.replace("_f16", "").replace("_f32", "")
        out_name = output_dir / f"{base_stem}_{q_type}.gguf"
        cmd = [quantize_bin, str(input_gguf), str(out_name), q_type]
        print(f"[-] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print(f"[✓] Created quantized model: {out_name} ({out_name.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"[!] Quantization failed for {q_type} (code {res.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Checkpoint to FUTO GGUF")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints/best_model",
                        help="Directory containing PyTorch model files and config.json")
    parser.add_argument("--tokenizer_file", type=str, default="models/tokenizer/tokenizer.model",
                        help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--output_file", type=str, default="models/gguf/el_keyboard_f16.gguf",
                        help="Path to output GGUF file")
    parser.add_argument("--languages", type=str, default="el",
                        help="Space-separated language codes (default: 'el')")
    parser.add_argument("--features", type=str, default="base_v1 inverted_space lora_finetunable_v1",
                        help="Space-separated feature flags (default: 'base_v1 inverted_space lora_finetunable_v1')")
    parser.add_argument("--out_type", type=str, choices=["f16", "f32"], default="f16",
                        help="Floating point precision for unquantized export")
    parser.add_argument("--quantize", action="store_true",
                        help="Automatically run llama-quantize to generate Q6_K and Q8_0")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    tokenizer_path = Path(args.tokenizer_file)
    output_path = Path(args.output_file)

    export_to_gguf(
        model_dir=model_path,
        tokenizer_file=tokenizer_path,
        output_file=output_path,
        out_type=args.out_type,
        languages=args.languages,
        features=args.features
    )

    if args.quantize:
        quantize_gguf(output_path, output_path.parent)


if __name__ == "__main__":
    main()
