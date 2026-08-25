# Greek Transformer Language Model for FUTO Keyboard

Custom Modern Greek (`el`) predictive text and autocorrect Transformer Language Model (~22.8M parameters) built for the **FUTO Android Keyboard**.

---

## 1. Overview & Architecture

This repository contains the complete end-to-end Python pipeline to gather datasets, train a SentencePiece tokenizer, train a compact LLaMA causal language model, inject FUTO custom metadata, and export/quantize GGUF models for mobile on-device inference.

```mermaid
flowchart TD
    subgraph S1["1. Data & QA Pipeline"]
        A["Data Sources: Opus-100, Opus Books, Greek Wikipedia"] --> B["01_download_and_clean_data.py"]
        B --> C["02_inspect_data.py Diagnostic Audit"]
    end

    subgraph S2["2. Tokenizer & Synthetic Corruptions"]
        C --> D["03_train_sentencepiece.py<br/>Vocab: 15,008 / treat_whitespace_as_suffix=True"]
        C --> E["04_generate_corruptions.py<br/>Accents 50%, Adjacency 30%, Homophones 10%, Typo 10%"]
    end

    subgraph S3["3. Training & Deployment"]
        D --> F["05_train_model.py<br/>22.8M Mini-LLaMA: 70% Clean + 30% Autocorrect"]
        E --> F
        F --> G["06_export_to_gguf.py<br/>Inject FUTO Metadata & Tokenizer Binary"]
        D --> G
        G --> H["07_evaluate_model.py<br/>Perplexity & Accent Benchmark"]
        G --> I["Quantize: Q6_K & Q8_0"]
        I --> J["Import into FUTO Keyboard APK"]
    end
```

### Model Specifications

| Parameter | Value | Description / Rationale |
| :--- | :--- | :--- |
| **Model Family** | `LlamaForCausalLM` | Supported by FUTO's embedded GGML runtime |
| **Parameters** | $\approx 22.8\text{M}$ | Real-time low latency (<20ms) on mobile CPUs |
| **Hidden Size ($d_{\text{model}}$)** | `512` | Standard FUTO mini-LLaMA dimension |
| **Hidden Layers** | `10` | Balances representation capacity and battery consumption |
| **Attention Heads** | `8` | Head dimension of 64 |
| **Intermediate Size** | `1376` | SwiGLU projection |
| **Max Context Window** | `256` tokens | Fits keyboard fast-forward context buffer |
| **Tokenizer** | SentencePiece (Unigram) | Required by FUTO runtime |
| **Tokenizer Flags** | `treat_whitespace_as_suffix=True` | **Mandatory** for keystroke prefix matching |
| **Vocabulary Size** | `15,008` tokens | Compact embedding table (~8MB) and fast softmax |
| **Control Tokens** | `<XBU>`, `<XBC>`, `<XEC>` | Autocorrect protocol: `[Ctx] <XBU>error<XBC>target<XEC>` |
| **Quantization** | `Q6_K` (~20MB) & `Q8_0` (~26MB) | Memory-efficient for mobile RAM |

---

---

## 2. Repository Structure

```text
greek-keyboard-lm/
├── 01_download_and_clean_data.py   # 1. Dataset streaming, cleaning, and QA splitting
├── 02_inspect_data.py              # 2. Diagnostic audit & monotonic accent compliance
├── 03_train_sentencepiece.py       # 3. SentencePiece Unigram trainer (15,008 vocab)
├── 04_generate_corruptions.py      # 4. Synthetic Greek typo & iotacism corruption engine
├── 05_train_model.py               # 5. 22.8M parameter mini-LLaMA PyTorch trainer
├── 06_export_to_gguf.py            # 6. GGUF exporter with embedded FUTO metadata & tokenizer
├── 07_evaluate_model.py            # 7. Benchmarks (PPL, Top-k, Accents) & Interactive REPL
├── models.json                     # FUTO Keyboard catalog manifest
├── requirements.txt                # Python dependencies
├── .env.example                    # Hugging Face authentication token template
└── README.md                       # Documentation
```

---

## 3. Installation & Setup

Ensure you are using Python 3.10+:

```bash
# 1. Clone repository
git clone https://github.com/stamchry/greek-keyboard-lm.git
cd greek-keyboard-lm

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Configure Hugging Face Token for faster streaming
cp .env.example .env
# Edit .env and paste your token: HF_TOKEN=hf_...
```

---

## 4. Pipeline Execution Guide

### Step 1: Download and Clean Greek Datasets
Ingests Opus-100 dialogues, Opus Books literature, and Greek Wikipedia. Strips subtitle timing cues, hearing-impaired tags, speaker markers, HTML, boilerplate, applies $\ge 85\%$ Greek character ratio filtering, normalizes Greek Unicode lookalikes (`ß` $\to$ `β`, `µ` $\to$ `μ`), and deduplicates lines.

```bash
python3 01_download_and_clean_data.py \
    --output_dir data/processed \
    --max_conv 200000 \
    --max_lit 30000 \
    --max_wiki 70000
```

Outputs:
- `data/processed/train.txt`
- `data/processed/val.txt`
- `data/processed/test.txt`
- `data/processed/dataset_stats.json`

---

### Step 2: Quality Assurance & Accent Audit
Audits token distributions, character ratios, monotonic accent compliance on polysyllabic words, and generates a structured Markdown report.

```bash
python3 02_inspect_data.py \
    --input_file data/processed/train.txt \
    --report_file data/qa_report.md
```

---

### Step 3: Train SentencePiece Tokenizer
Trains a 15,008 vocabulary Unigram tokenizer with `treat_whitespace_as_suffix=True` and FUTO control tokens (`<XBU>`, `<XBC>`, `<XEC>`).

```bash
python3 03_train_sentencepiece.py \
    --input_file data/processed/train.txt \
    --output_dir models/tokenizer \
    --vocab_size 15008
```

Outputs:
- `models/tokenizer/tokenizer.model`
- `models/tokenizer/tokenizer.vocab`
- Hugging Face configuration files (`tokenizer_config.json`, `special_tokens_map.json`)

---

### Step 4: Generate Synthetic Autocorrect Benchmarks
Generates synthetic Greek typing errors for test evaluation:
- 50% Accent stripping (e.g. `καλημερα` $\to$ `καλημέρα`)
- 30% Greek QWERTY adjacency (e.g. `α` $\leftrightarrow$ `σ`, `φ` $\leftrightarrow$ `γ`, `θ` $\leftrightarrow$ `ι`)
- 10% Homophones / Iotacism (e.g. `/i/`: `ι` $\leftrightarrow$ `η` $\leftrightarrow$ `υ` $\leftrightarrow$ `ει` $\leftrightarrow$ `οι`, `/e/`: `ε` $\leftrightarrow$ `αι`)
- 10% Typo dynamics (transpositions, deletions, insertions)

```bash
python3 04_generate_corruptions.py \
    --input_file data/processed/test.txt \
    --output_file data/autocorrect_eval.jsonl \
    --num_samples 1000
```

---

### Step 5: Train 22M LLaMA Model
Trains the custom 22M parameter LLaMA model using PyTorch, dynamically mixing 70% clean next-token sequences and 30% autocorrect prompts with AdamW and cosine decay.

```bash
python3 05_train_model.py \
    --train_file data/processed/train.txt \
    --val_file data/processed/val.txt \
    --tokenizer_dir models/tokenizer \
    --output_dir models/checkpoints \
    --epochs 5 \
    --batch_size 32 \
    --lr 3e-4
```

---

### Step 6: Export to GGUF & Inject FUTO Metadata
Converts trained PyTorch weights to GGUF format and embeds the raw SentencePiece binary and required FUTO metadata:

```bash
python3 06_export_to_gguf.py \
    --model_dir models/checkpoints/best_model \
    --tokenizer_file models/tokenizer/tokenizer.model \
    --output_file models/gguf/el_keyboard_f16.gguf \
    --quantize
```

Injected FUTO Metadata Keys:
```python
keyboardlm.languages = "el"
keyboardlm.features = "base_v1 inverted_space xbu_char_autocorrect_v1 lora_finetunable_v1"
keyboardlm.ext_tokenizer_type = "sentencepiece"
keyboardlm.ext_tokenizer_data = <raw tokenizer.model binary>
keyboardlm.finetuning_count = 0
keyboardlm.history = ""
```

---

### Step 7: Evaluate Model Performance
Runs next-token perplexity, top-k accuracy, and verifies the $>95\%$ accent restoration benchmark. Also includes an interactive console mode:

```bash
# Automated evaluation benchmark
python3 07_evaluate_model.py \
    --model_dir models/checkpoints/best_model \
    --tokenizer_file models/tokenizer/tokenizer.model \
    --test_file data/processed/test.txt \
    --eval_jsonl data/autocorrect_eval.jsonl

# Interactive live console mode
python3 07_evaluate_model.py \
    --model_dir models/checkpoints/best_model \
    --tokenizer_file models/tokenizer/tokenizer.model \
    --interactive
```

---

## 4. Mobile Deployment to FUTO Keyboard

1. **Copy Model**: Transfer `el_keyboard_Q6_K.gguf` to your Android device storage.
2. **Open FUTO Keyboard**:
   - Go to **Settings → Predictive Text → Transformer Models**.
   - Tap **Actions → Import from file**.
   - Select `el_keyboard_Q6_K.gguf`.
3. **Verify**:
   - Enable Greek keyboard layout.
   - Type unaccented words (e.g. `καλημερα`) and observe immediate accented suggestions (`καλημέρα`).

---

## 5. License

Apache-2.0 License.
