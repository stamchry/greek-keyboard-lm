#!/usr/bin/env python3
"""
08_upload_to_hf.py - Upload Greek Keyboard LM & GGUF Artifacts to Hugging Face Hub
"""

import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Load .env token if present
env_path = Path('.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith('HF_TOKEN='):
            token_val = line.split('=', 1)[1].strip().strip('\"\'')
            if token_val:
                os.environ['HF_TOKEN'] = token_val
            break

api = HfApi()
user_info = api.whoami()
username = user_info.get("name", user_info.get("username"))
repo_id = f"{username}/greek-keyboard-lm"

print(f"[-] Authenticated as: {username}")
print(f"[-] Target Repository: https://huggingface.co/{repo_id}")

# 1. Create repo if it does not exist
repo_url = create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"[✓] Repository ready: {repo_url}")

# 2. Stage files in a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    stage_path = Path(temp_dir)
    
    # A. Copy PyTorch & Configs
    best_model_dir = Path("models/checkpoints/best_model")
    if (best_model_dir / "model.safetensors").exists():
        shutil.copy(best_model_dir / "model.safetensors", stage_path / "model.safetensors")
        shutil.copy(best_model_dir / "config.json", stage_path / "config.json")
        if (best_model_dir / "generation_config.json").exists():
            shutil.copy(best_model_dir / "generation_config.json", stage_path / "generation_config.json")
            
    # B. Copy Tokenizer
    tok_dir = Path("models/tokenizer")
    for f in ["tokenizer.model", "tokenizer.vocab", "tokenizer_config.json", "special_tokens_map.json"]:
        if (tok_dir / f).exists():
            shutil.copy(tok_dir / f, stage_path / f)
            
    # C. Copy GGUF Models (latest v3)
    gguf_dir = Path("models/gguf")
    gguf_files = {
        "el_keyboard_v3_Q6_K.gguf": "el_keyboard_v3_Q6_K.gguf",
        "el_keyboard_v3_Q8_0.gguf": "el_keyboard_v3_Q8_0.gguf",
        "el_keyboard_v3_f16.gguf": "el_keyboard_v3_f16.gguf",
    }
    for src_name, dst_name in gguf_files.items():
        src_path = gguf_dir / src_name
        if src_path.exists():
            shutil.copy(src_path, stage_path / dst_name)
            # Also create standard alias names for easy direct downloading
            alias_name = dst_name.replace("_v3", "")
            shutil.copy(src_path, stage_path / alias_name)

    # D. Model Card (README.md for Hugging Face)
    model_card = f"""---
language:
- el
license: apache-2.0
tags:
- futo
- keyboard
- gguf
- text-generation
- casual-lm
- mobile
pipeline_tag: text-generation
---

# Greek Transformer Language Model for FUTO Keyboard (~36.1M)

Custom Modern Greek (`el`) predictive text and autocorrect Transformer Language Model (~36.15M parameters) built for the **FUTO Android Keyboard**.

- **Architecture:** `LlamaForCausalLM` (9 layers, 512 hidden size, 1376 intermediate, 8 heads, tied embeddings)
- **Quantization Formats:** `Q6_K` (~36.8 MB), `Q8_0` (~45.4 MB), `F16` (~85 MB)
- **Tokenizer:** SentencePiece Unigram (`15,008` vocab, `treat_whitespace_as_suffix=True`)
- **Control Tokens:** `<XBU>`, `<XBC>`, `<XEC>` for FUTO autocorrect protocol

## 🚀 Quick Start & Mobile Deployment

### 1. Requirements
To support Greek in-word typing, Unicode codepoints, and spatial keystroke decoding, install the APK built from the Greek-enabled fork:
👉 **[https://github.com/stamchry/android-keyboard](https://github.com/stamchry/android-keyboard)**

### 2. Download Model
Download `el_keyboard_v3_Q6_K.gguf` (or `el_keyboard_Q6_K.gguf`) to your Android device storage.

### 3. Setup in FUTO Keyboard
1. Open **FUTO Keyboard Settings → Help & About**.
2. Tap **"Version code" 8 times rapidly** to unlock Developer Settings.
3. Open **Settings → Developer** → toggle **"Allow transformer models on non QWERTY layouts"** to **ON**.
4. Go to **Settings → Predictive Text → Transformer Models**.
5. Tap **Actions (top-right) → Import from file** → select `el_keyboard_v3_Q6_K.gguf`.
6. Ensure it displays: **"Model is set to default for el"**.

---

## 📊 Benchmark Results

| Metric / Benchmark | Result |
| :--- | :--- |
| **Next-Word Top-1 Accuracy** | **39.33%** |
| **Next-Word Top-3 Accuracy** | **53.06%** |
| **Next-Word Top-10 Accuracy** | **64.48%** |
| **Accent Restoration Accuracy** | **38.60%** |
| **Synthetic Typo Autocorrect** | **37.40%** |

---

## 🔗 Repository
Source training code and pipeline: [https://github.com/stamchry/greek-keyboard-lm](https://github.com/stamchry/greek-keyboard-lm)
"""
    (stage_path / "README.md").write_text(model_card, encoding="utf-8")
    
    print(f"[-] Uploading all staged artifacts to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=str(stage_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: upload Greek Keyboard LM v3 PyTorch weights, GGUFs (Q6_K, Q8_0, F16), and tokenizer"
    )
    print(f"[✓] Upload completed successfully! View your model at:")
    print(f"    https://huggingface.co/{repo_id}")

