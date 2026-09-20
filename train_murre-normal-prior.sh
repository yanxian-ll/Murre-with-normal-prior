#!/usr/bin/env bash
# Fine-tune local Murre with frozen, online Metric3D RGB -> normal predictions.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.tools/murre-env/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/nas/Mapanything_dataset/datasets_processed}"
INDEX_PATH="${INDEX_PATH:-$ROOT/dataset/murre_normal_training_pairs.txt}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints/murre}"
METRIC3D_CHECKPOINT="${METRIC3D_CHECKPOINT:-$ROOT/checkpoints/Metric3D/metric_depth_vit_large_800k.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/murre-normal-prior-b16}"
RESOLUTIONS="${RESOLUTIONS:-192x256 256x384}" # HxW, same size within each batch
# Resize RGB/depth together before cropping; K follows both transforms.
WORK_RESOLUTION="${WORK_RESOLUTION:-384}" # longest edge before crop
CROP_SCALE_MIN="${CROP_SCALE_MIN:-1}" # fraction of largest aspect-matched crop
CROP_SCALE_MAX="${CROP_SCALE_MAX:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NORMAL_WEIGHT="${NORMAL_WEIGHT:-0.1}"
NORMAL_KEEP_RATIO="${NORMAL_KEEP_RATIO:-0.9}"
METRIC3D_DEVICE="${METRIC3D_DEVICE:-cuda}"
METRIC3D_MAX_EDGE="${METRIC3D_MAX_EDGE:-1064}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_RETRIES="${MAX_RETRIES:-1000}"
MAX_DEPTH="${MAX_DEPTH:-0}" # 0: no absolute cap; use each frame's 99.5th percentile * 1.5
PRECISION="${PRECISION:-bf16}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-20}"
AUTO_RESUME="${AUTO_RESUME:-1}"
VAL_ON_START="${VAL_ON_START:-1}"
VAL_EVERY="${VAL_EVERY:-500}" # 0 disables
VAL_SAMPLES="${VAL_SAMPLES:-20}"
VAL_RESOLUTION="${VAL_RESOLUTION:-192x256}"
VAL_DENOISING_STEPS="${VAL_DENOISING_STEPS:-4}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
read -r -a sizes <<< "$RESOLUTIONS"
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
build=("$PYTHON_BIN" "$ROOT/dataset/build_murre_training_index.py" --root "$DATA_ROOT" --output "$INDEX_PATH")
if [[ ! -s "$INDEX_PATH" || "${REBUILD_INDEX:-0}" == 1 ]]; then
  printf '构建索引: '; printf '%q ' "${build[@]}"; printf '\n'
  if ! "$dry_run"; then "${build[@]}"; fi
fi
cmd=("$PYTHON_BIN" "$ROOT/Murre-with-normal-prior/train.py"
  --index "$INDEX_PATH" --checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR"
  --normal_source metric3d --metric3d_checkpoint "$METRIC3D_CHECKPOINT"
  --metric3d_device "$METRIC3D_DEVICE" --metric3d_max_edge "$METRIC3D_MAX_EDGE"
  --work_resolution "$WORK_RESOLUTION" --crop_scale_min "$CROP_SCALE_MIN" --crop_scale_max "$CROP_SCALE_MAX"
  --resolutions "${sizes[@]}" --batch_size "$BATCH_SIZE" --gradient_accumulation_steps "$GRAD_ACCUM"
  --max_steps "$MAX_STEPS" --learning_rate "$LEARNING_RATE" --num_workers "$NUM_WORKERS"
  --max_retries "$MAX_RETRIES" --max_depth_mode auto --max_depth "$MAX_DEPTH"
  --normal_weight "$NORMAL_WEIGHT" --normal_keep_ratio "$NORMAL_KEEP_RATIO"
  --precision "$PRECISION" --gradient_checkpointing --random_flip --save_every "$SAVE_EVERY"
  --log_every "$LOG_EVERY" --tensorboard_dir "$TENSORBOARD_DIR"
  --val_every "$VAL_EVERY" --val_samples "$VAL_SAMPLES" --val_resolution "$VAL_RESOLUTION"
  --val_denoising_steps "$VAL_DENOISING_STEPS")
if [[ -n "${VAL_INDEX:-}" ]]; then cmd+=(--val_index "$VAL_INDEX"); fi
if [[ -z "${RESUME:-}" && "$AUTO_RESUME" == 1 ]]; then
  RESUME="$("$PYTHON_BIN" - "$OUTPUT_DIR" "$ROOT" <<'PYRESUME'
import sys
from pathlib import Path
out, root = Path(sys.argv[1]), Path(sys.argv[2])
candidates = list(out.glob('checkpoint-*'))
if (out/'final').is_dir():
    candidates.append(out/'final')
latest = out/'latest.txt'
if latest.is_file():
    value = latest.read_text().strip()
    if value:
        p = Path(value)
        candidates.extend([p, root/p, out/p.name])
def complete(p):
    return ((p/'trainer_state.pt').is_file() and (p/'model_index.json').is_file()
            and (any((p/'unet').glob('*.safetensors')) or any((p/'unet').glob('*.bin'))))
valid = {p.resolve() for p in candidates if complete(p)}
# Modification time also permits a completed final checkpoint.
if valid:
    print(max(valid, key=lambda p:(p/'trainer_state.pt').stat().st_mtime))
elif candidates or latest.exists():
    raise SystemExit('Found checkpoint records but no complete checkpoint; check saved files or set AUTO_RESUME=0 explicitly.')
PYRESUME
)"
fi
if [[ -n "${RESUME:-}" ]]; then
  printf '恢复 checkpoint: %s\n' "$RESUME"
  cmd+=(--resume "$RESUME")
else
  printf '从预训练权重开始新训练\n'
fi
if [[ "$VAL_ON_START" == 1 ]]; then cmd+=(--val_on_start); fi
cmd+=("$@")
printf '训练命令: '; printf '%q ' "${cmd[@]}"; printf '\n'
if "$dry_run"; then exit 0; fi
exec "${cmd[@]}"
