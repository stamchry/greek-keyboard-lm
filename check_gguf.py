#!/usr/bin/env python3
"""
check_gguf.py - Comprehensive GGUF and FUTO KeyboardLM Metadata Inspector
"""

import sys
import io
from pathlib import Path
import gguf
import sentencepiece as spm


def inspect_gguf(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"\033[1;31m[!] Error: File not found: {file_path}\033[0m")
        return

    print("=" * 70)
    print(f"\033[1;36mInspecting GGUF Model: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)\033[0m")
    print("=" * 70)

    reader = gguf.GGUFReader(str(path))

    print("\n\033[1;33m[1] FUTO KeyboardLM Custom Metadata:\033[0m")
    futo_keys = [k for k in reader.fields.keys() if k.startswith("keyboardlm.")]
    if not futo_keys:
        print("  \033[1;31m[!] NO keyboardlm.* metadata found!\033[0m")
    else:
        for k in futo_keys:
            field = reader.fields[k]
            if k == "keyboardlm.ext_tokenizer_data":
                data_bytes = bytes([int(field.parts[idx][0]) for idx in field.data]) if field.data else b""
                print(f"  • \033[1;32m{k}\033[0m: <Embedded Binary, {len(data_bytes)} bytes>")
                # Verify SentencePiece loading from embedded bytes
                try:
                    sp = spm.SentencePieceProcessor()
                    sp.LoadFromSerializedProto(data_bytes)
                    print(f"    └─ \033[1;32m[✓] Embedded SentencePiece Validated! Vocab size: {sp.GetPieceSize()}\033[0m")
                    # Check control tokens
                    xbu_id = sp.PieceToId("<XBU>")
                    xbc_id = sp.PieceToId("<XBC>")
                    xec_id = sp.PieceToId("<XEC>")
                    print(f"    └─ Control Tokens: <XBU>={xbu_id}, <XBC>={xbc_id}, <XEC>={xec_id}")
                except Exception as e:
                    print(f"    └─ \033[1;31m[!] Failed to parse embedded tokenizer proto: {e}\033[0m")
            else:
                try:
                    raw_part = field.parts[field.data[0]] if field.data else field.parts[0]
                    if isinstance(raw_part, (bytes, bytearray)):
                        val = bytes(raw_part).decode('utf-8', errors='replace')
                    elif isinstance(raw_part, (list, tuple)) or hasattr(raw_part, '__iter__'):
                        val = bytes(raw_part).decode('utf-8', errors='replace')
                    else:
                        val = str(raw_part)
                except Exception:
                    val = str(field)
                print(f"  • \033[1;32m{k}\033[0m: {val}")

    print("\n\033[1;33m[2] General & Architecture Metadata:\033[0m")
    arch_keys = [k for k in reader.fields.keys() if not k.startswith("keyboardlm.") and not k.startswith("tokenizer.ggml.tokens")]
    for k in arch_keys:
        field = reader.fields[k]
        if "token" in k and len(field.data) > 10:
            print(f"  • {k}: <Array of {len(field.data)} elements>")
        else:
            try:
                if field.data and len(field.data) == 1:
                    part = field.parts[field.data[0]]
                    val = bytes(part).decode('utf-8', errors='replace') if isinstance(part, (bytes, bytearray)) else part[0]
                else:
                    val = f"<Array of {len(field.data)} items>"
            except Exception:
                val = str(field)
            print(f"  • {k}: {val}")

    print(f"\n\033[1;33m[3] Tensor Summary:\033[0m")
    print(f"  • Total Tensors in GGUF: {len(reader.tensors)}")
    dtypes = {}
    for t in reader.tensors:
        dt = str(t.tensor_type)
        dtypes[dt] = dtypes.get(dt, 0) + 1
    print(f"  • Tensor Types: {dtypes}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_path = "models/gguf/el_keyboard_Q6_K.gguf"
    else:
        model_path = sys.argv[1]
    inspect_gguf(model_path)
