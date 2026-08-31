#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Greek Keyboard LM: Training & GGUF Pipeline"
echo "============================================================"

# Activate Conda Environment
eval "$(conda shell.bash hook)"
conda activate ml

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p models/checkpoints models/gguf

echo "[1/4] Starting Model Training on CUDA..."
python3 05_train_model.py \
    --train_file data/processed/train.txt \
    --val_file data/processed/val.txt \
    --tokenizer_dir models/tokenizer \
    --output_dir models/checkpoints \
    --epochs 6 \
    --batch_size 64 \
    --grad_accum 2 \
    --lr 5e-4 \
    --warmup_steps 500 \
    --eval_every 500 \
    --save_every 1000 \
    --max_seq_len 256 \
    --autocorrect_ratio 0.50

echo "[2/4] Exporting and Quantizing to GGUF (v3)..."
python3 06_export_to_gguf.py \
    --model_dir models/checkpoints/best_model \
    --tokenizer_file models/tokenizer/tokenizer.model \
    --output_file models/gguf/el_keyboard_v3_f16.gguf \
    --quantize

echo "[3/4] Inspecting & Validating Exported Model..."
python3 check_gguf.py models/gguf/el_keyboard_v3_Q6_K.gguf

echo "[4/4] Evaluating Model Benchmarks..."
python3 07_evaluate_model.py \
    --model_dir models/checkpoints/best_model \
    --tokenizer_file models/tokenizer/tokenizer.model \
    --test_file data/processed/test.txt

echo "============================================================"
echo "Pipeline Completed Successfully!"
echo "New models ready at: models/gguf/el_keyboard_v3_Q6_K.gguf"
echo "============================================================"
