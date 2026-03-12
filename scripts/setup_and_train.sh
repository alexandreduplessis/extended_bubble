#!/usr/bin/env bash
# Full cluster setup: download MSD dataset, preprocess, and train.
#
# Usage:
#   bash scripts/setup_and_train.sh [DATA_DIR] [CHECKPOINT_DIR]
#
# DATA_DIR       defaults to /Data/$USER
# CHECKPOINT_DIR defaults to DATA_DIR/checkpoints
#
# Expects: CUDA GPU, kaggle credentials (~/.kaggle/kaggle.json), wandb login.

set -euo pipefail

DATA_DIR="${1:-/Data/$USER}"
CHECKPOINT_DIR="${2:-$DATA_DIR/checkpoints}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Extended Bubble: Cluster Setup ==="
echo "Repo:        $REPO_DIR"
echo "Data dir:    $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# ── 1. Python environment ─────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv "$REPO_DIR/.venv"
fi
source "$REPO_DIR/.venv/bin/activate"

echo ">>> Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$REPO_DIR/requirements.txt" -q

# Install the project itself so `from src.*` imports work
pip install -e "$REPO_DIR" -q

# ── 2. Download MSD dataset ───────────────────────────────────────────
MSD_DIR="$DATA_DIR/msd"
if [ -d "$MSD_DIR/modified-swiss-dwellings-v2" ]; then
    echo ">>> MSD dataset already present at $MSD_DIR, skipping download."
else
    echo ">>> Downloading MSD dataset via kaggle..."
    mkdir -p "$MSD_DIR"
    kaggle datasets download -d simontremblay/modified-swiss-dwellings-v2 \
        -p "$MSD_DIR" --unzip
    echo ">>> Dataset downloaded to $MSD_DIR"
fi

# ── 3. Preprocess (extract bubbles/boundaries to .npz) ────────────────
CACHE_DIR="$DATA_DIR/msd_preprocessed"
TRAIN_CACHE="$CACHE_DIR/train"

# Check if preprocessing looks complete (compare file counts)
GRAPH_COUNT=$(find "$MSD_DIR/modified-swiss-dwellings-v2/train/graph_out" -name "*.pickle" 2>/dev/null | wc -l)
CACHE_COUNT=$(find "$TRAIN_CACHE" -name "*.npz" 2>/dev/null | wc -l)

if [ "$CACHE_COUNT" -ge "$GRAPH_COUNT" ] && [ "$GRAPH_COUNT" -gt 0 ]; then
    echo ">>> Preprocessing already complete ($CACHE_COUNT/$GRAPH_COUNT files), skipping."
else
    echo ">>> Preprocessing dataset ($CACHE_COUNT/$GRAPH_COUNT done so far)..."
    cd "$REPO_DIR"
    python scripts/preprocess.py "$MSD_DIR" "$CACHE_DIR"
    echo ">>> Preprocessing complete."
fi

# ── 4. Update config paths ────────────────────────────────────────────
mkdir -p "$CHECKPOINT_DIR"
RUNTIME_CONFIG="$REPO_DIR/configs/runtime.yaml"
cat > "$RUNTIME_CONFIG" << EOF
data:
  msd_path: $MSD_DIR
  n_max: 135
  n_boundary_max: 64
  n_constraints_max: 300
  num_room_types: 9
  num_constraint_types: 11
  constraint_dim: 30
  augment_rotations: true
  augment_flips: true
  min_constraints: 10
  max_constraints: 200

model:
  d_model: 256
  n_layers: 8
  n_heads: 8
  type_emb_dim: 8
  slot_dim: 11
  ffn_ratio: 4
  dropout: 0.0

diffusion:
  num_timesteps: 1000
  schedule: cosine
  type_schedule_shift: 1.0
  prediction: epsilon

training:
  batch_size: 128
  lr: 1.0e-4
  warmup_steps: 1000
  max_steps: 200000
  val_every: 5000
  save_every: 50
  cfg_drop_single: 0.1
  cfg_drop_all: 0.05
  cfg_drop_boundary: 0.02
  type_loss_weight: 2.0
  grad_clip: 1.0
  checkpoint_dir: $CHECKPOINT_DIR

inference:
  num_steps: 100
  cfg_scale: 3.0
  energy_guidance_lambda: 0.0

eval:
  num_samples: 500
EOF

echo ">>> Generated runtime config at $RUNTIME_CONFIG"

# ── 5. Wandb login ────────────────────────────────────────────────────
if ! wandb status 2>/dev/null | grep -q "Logged in"; then
    echo ">>> Please log in to wandb:"
    wandb login
fi

# ── 6. Train ──────────────────────────────────────────────────────────
echo ""
echo "=== Starting training ==="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Wandb project: extendedbubble"
echo ""

cd "$REPO_DIR"
python scripts/train.py configs/runtime.yaml
