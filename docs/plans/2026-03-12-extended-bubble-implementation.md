# Extended Bubble Graph Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a slot-based diffusion transformer that generates room bubbles (center, radius, type) from a building boundary + architectural constraints, trained on MSD.

**Architecture:** Transformer denoiser on N_max padded slots with AdaLN-Zero timestep conditioning, cross-attention to boundary vertices and constraint tokens. DDPM training with classifier-free guidance via constraint dropout.

**Tech Stack:** Python 3.11, PyTorch 2.x, shapely, networkx, numpy, matplotlib, wandb (logging), kaggle API (download)

---

## Task 1: Project Setup

**Files:**
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/pyproject.toml`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/data/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/model/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/training/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/inference/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/src/evaluation/__init__.py`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/configs/default.yaml`
- Create: `/users/eleves-a/2022/amine.chraibi/extended_bubble/scripts/download_msd.sh`

**Step 1: Create virtual environment**

```bash
cd /Data/amine.chraibi
python3 -m venv extended_bubble_env
source extended_bubble_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install shapely networkx numpy matplotlib pyyaml wandb kaggle tqdm scipy
```

**Step 2: Create project structure and pyproject.toml**

```toml
[project]
name = "extended-bubble"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

**Step 3: Create default config**

```yaml
# configs/default.yaml
data:
  msd_path: /Data/amine.chraibi/msd
  n_max: 200  # to be refined after data exploration
  n_boundary_max: 64
  n_constraints_max: 300
  num_room_types: 9  # Bedroom(0)..Balcony(8)
  num_constraint_types: 11
  constraint_dim: 30  # 11 + 9 + 9 + 1
  augment_rotations: true
  augment_flips: true
  min_constraints: 10
  max_constraints: 200

model:
  d_model: 256
  n_layers: 8
  n_heads: 8
  type_emb_dim: 8
  slot_dim: 11  # x, y, r, type_emb(8)
  ffn_ratio: 4
  dropout: 0.0

diffusion:
  num_timesteps: 1000
  schedule: cosine
  type_schedule_shift: 0.3  # types clean by 30% of process
  prediction: epsilon  # epsilon or v

training:
  batch_size: 32
  lr: 1.0e-4
  warmup_steps: 1000
  max_steps: 200000
  val_every: 5000
  save_every: 10000
  cfg_drop_single: 0.1   # drop each constraint with p=0.1
  cfg_drop_all: 0.05     # drop all constraints with p=0.05
  cfg_drop_boundary: 0.02
  type_loss_weight: 2.0
  grad_clip: 1.0

inference:
  num_steps: 100  # DDIM steps
  cfg_scale: 3.0
  energy_guidance_lambda: 0.0  # 0 = disabled

eval:
  num_samples: 500
```

**Step 4: Create download script**

```bash
#!/bin/bash
# scripts/download_msd.sh
mkdir -p /Data/amine.chraibi/msd
cd /Data/amine.chraibi/msd
kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings
unzip -o modified-swiss-dwellings.zip
rm modified-swiss-dwellings.zip
```

**Step 5: Initialize git repo and commit**

```bash
cd /users/eleves-a/2022/amine.chraibi/extended_bubble
git init
git add pyproject.toml configs/ src/ scripts/
git commit -m "feat: project scaffold with config and directory structure"
```

---

## Task 2: Data Exploration — Determine N_max and Statistics

**Files:**
- Create: `scripts/explore_msd.py`

**Step 1: Write exploration script**

```python
"""Explore MSD dataset to determine N_max and collect statistics."""
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import sys

MSD_PATH = Path("/Data/amine.chraibi/msd")

def explore():
    train_graph_out = MSD_PATH / "modified-swiss-dwellings-v1-train" / "graph_out"
    if not train_graph_out.exists():
        # try alternate structure
        train_graph_out = MSD_PATH / "graph_out"

    pickle_files = sorted(train_graph_out.glob("*.pickle"))
    print(f"Found {len(pickle_files)} floor plans")

    room_counts = []
    type_counts = Counter()
    areas = []
    radii = []

    for pf in pickle_files:
        with open(pf, "rb") as f:
            G = pickle.load(f)

        n_rooms = 0
        for node_id, data in G.nodes(data=True):
            rt = data.get("room_type", -1)
            if rt in range(9):  # 0..8, exclude Structure(9) and Background(13)
                n_rooms += 1
                type_counts[rt] += 1
                geom = data.get("geometry")
                if geom is not None:
                    a = geom.area
                    areas.append(a)
                    radii.append(np.sqrt(a / np.pi))
        room_counts.append(n_rooms)

    room_counts = np.array(room_counts)
    print(f"\nRoom count stats:")
    print(f"  min={room_counts.min()}, max={room_counts.max()}, "
          f"mean={room_counts.mean():.1f}, median={np.median(room_counts):.1f}")
    print(f"  p95={np.percentile(room_counts, 95):.0f}, "
          f"p99={np.percentile(room_counts, 99):.0f}")

    print(f"\nType distribution:")
    type_names = ["Bedroom", "Living", "Kitchen", "Dining", "Corridor",
                  "Stairs", "Storeroom", "Bathroom", "Balcony"]
    for i in range(9):
        print(f"  {type_names[i]}: {type_counts[i]}")

    areas = np.array(areas)
    radii = np.array(radii)
    print(f"\nArea stats (m^2): min={areas.min():.2f}, max={areas.max():.2f}, "
          f"mean={areas.mean():.2f}, median={np.median(areas):.2f}")
    print(f"Radius stats (m): min={radii.min():.2f}, max={radii.max():.2f}, "
          f"mean={radii.mean():.2f}, median={np.median(radii):.2f}")

    print(f"\nRecommended N_max (p99 + 10% buffer): "
          f"{int(np.percentile(room_counts, 99) * 1.1)}")

if __name__ == "__main__":
    explore()
```

**Step 2: Run and record N_max**

```bash
source /Data/amine.chraibi/extended_bubble_env/bin/activate
python scripts/explore_msd.py
```

Update `configs/default.yaml` with the actual N_max value.

**Step 3: Commit**

```bash
git add scripts/explore_msd.py
git commit -m "feat: MSD exploration script for N_max and statistics"
```

---

## Task 3: Bubble Extraction from MSD

**Files:**
- Create: `src/data/bubble_extractor.py`
- Create: `tests/test_bubble_extractor.py`

**Step 1: Write failing test**

```python
# tests/test_bubble_extractor.py
import numpy as np
from shapely.geometry import Polygon
import networkx as nx
from src.data.bubble_extractor import extract_bubbles, extract_boundary

def make_mock_graph():
    """Create a minimal MSD-like graph with 3 rooms."""
    G = nx.Graph()
    # Bedroom: 4x3 rectangle at (2, 1.5)
    G.add_node(0, room_type=0,
               centroid=(2.0, 1.5),
               geometry=Polygon([(0,0),(4,0),(4,3),(0,3)]))
    # Kitchen: 3x3 square at (5.5, 1.5)
    G.add_node(1, room_type=2,
               centroid=(5.5, 1.5),
               geometry=Polygon([(4,0),(7,0),(7,3),(4,3)]))
    # Bathroom: 2x2 square at (1, 4)
    G.add_node(2, room_type=7,
               centroid=(1.0, 4.0),
               geometry=Polygon([(0,3),(2,3),(2,5),(0,5)]))
    # Structure node (should be excluded)
    G.add_node(3, room_type=9,
               centroid=(3.5, 4.0),
               geometry=Polygon([(2,3),(5,3),(5,5),(2,5)]))
    G.add_edge(0, 1, connectivity="door")
    G.add_edge(0, 2, connectivity="passage")
    return G

def test_extract_bubbles():
    G = make_mock_graph()
    bubbles = extract_bubbles(G)
    # Should have 3 rooms (exclude Structure)
    assert len(bubbles) == 3
    # Each bubble: (cx, cy, radius, room_type)
    assert all(len(b) == 4 for b in bubbles)
    # Check bedroom radius ~= sqrt(12/pi)
    bedroom = [b for b in bubbles if b[3] == 0][0]
    expected_r = np.sqrt(12.0 / np.pi)
    assert abs(bedroom[2] - expected_r) < 0.01
    # Check types
    types = sorted([b[3] for b in bubbles])
    assert types == [0, 2, 7]

def test_extract_boundary():
    struct_in = np.zeros((512, 512, 3), dtype=np.float16)
    # Create a simple rectangular structure mask
    struct_in[100:400, 100:400, 0] = 0  # 0 = structure
    struct_in[:100, :, 0] = 1           # 1 = non-structure
    struct_in[400:, :, 0] = 1
    struct_in[:, :100, 0] = 1
    struct_in[:, 400:, 0] = 1
    # Set real-world coordinates
    for i in range(512):
        struct_in[i, :, 1] = np.linspace(0, 10, 512)  # x
    for j in range(512):
        struct_in[:, j, 2] = np.linspace(0, 10, 512)  # y

    boundary = extract_boundary(struct_in)
    assert len(boundary) >= 4  # at least 4 vertices for a rectangle
    # Check boundary is roughly a 0-10 scale
    assert boundary.shape[1] == 2
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_bubble_extractor.py -v
```

**Step 3: Write implementation**

```python
# src/data/bubble_extractor.py
"""Extract bubble representations and boundaries from MSD floor plans."""
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import networkx as nx
from typing import List, Tuple

VALID_ROOM_TYPES = set(range(9))  # 0..8


def extract_bubbles(graph: nx.Graph) -> List[Tuple[float, float, float, int]]:
    """Extract (cx, cy, radius, room_type) for each valid room in the graph."""
    bubbles = []
    for _, data in graph.nodes(data=True):
        room_type = data.get("room_type", -1)
        if room_type not in VALID_ROOM_TYPES:
            continue
        geom = data.get("geometry")
        if geom is None:
            continue
        centroid = geom.centroid
        area = geom.area
        radius = np.sqrt(area / np.pi)
        bubbles.append((centroid.x, centroid.y, radius, int(room_type)))
    return bubbles


def extract_boundary(struct_in: np.ndarray, simplify_tolerance: float = 0.1) -> np.ndarray:
    """Extract boundary polygon vertices from struct_in array.

    struct_in shape: (512, 512, 3)
      channel 0: binary mask (0=structure, 1=non-structure)
      channel 1: x-coordinate in meters
      channel 2: y-coordinate in meters

    Returns: (N, 2) array of boundary vertices in meter coordinates.
    """
    # Structure mask: 0 = structure pixels
    structure_mask = (struct_in[:, :, 0] < 0.5).astype(np.uint8)

    if structure_mask.sum() == 0:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    # Find contour of structure using shapely on the pixel coords
    from scipy import ndimage
    # Fill holes in structure
    filled = ndimage.binary_fill_holes(structure_mask)

    # Convert pixel mask to polygon using marching squares
    from shapely.geometry import MultiPoint

    # Get pixel coordinates of filled structure
    ys, xs = np.where(filled)
    if len(xs) == 0:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    # Map pixel coords to real-world meters
    x_coords = struct_in[:, :, 1]
    y_coords = struct_in[:, :, 2]

    # Get convex hull of structure pixels in real-world coordinates
    real_x = x_coords[ys, xs].astype(np.float64)
    real_y = y_coords[ys, xs].astype(np.float64)

    points = np.column_stack([real_x, real_y])

    # Use alpha shape or convex hull
    from shapely.geometry import MultiPoint
    mp = MultiPoint(points[::10])  # subsample for speed
    hull = mp.convex_hull

    if hull.is_empty:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    simplified = hull.simplify(simplify_tolerance)
    coords = np.array(simplified.exterior.coords)[:-1]  # remove closing duplicate
    return coords.astype(np.float32)
```

**Step 4: Run tests, iterate until passing**

```bash
pytest tests/test_bubble_extractor.py -v
```

**Step 5: Commit**

```bash
git add src/data/bubble_extractor.py tests/test_bubble_extractor.py
git commit -m "feat: bubble and boundary extraction from MSD graphs"
```

---

## Task 4: Constraint System

**Files:**
- Create: `src/data/constraints.py`
- Create: `tests/test_constraints.py`

**Step 1: Write failing test**

```python
# tests/test_constraints.py
import numpy as np
from src.data.constraints import (
    ConstraintType, Constraint, encode_constraint,
    generate_constraints_from_bubbles
)

def test_encode_constraint():
    c = Constraint(
        type=ConstraintType.MIN_AREA,
        room_type_a=0,  # Bedroom
        room_type_b=-1,  # unused
        value=9.0
    )
    vec = encode_constraint(c)
    assert vec.shape == (30,)
    # type_onehot: 11 dims, first is MIN_AREA=0
    assert vec[0] == 1.0
    # room_type_a onehot: 9 dims starting at idx 11
    assert vec[11] == 1.0  # Bedroom=0
    # value at idx 29
    assert vec[29] == 9.0

def test_generate_constraints():
    bubbles = [
        (2.0, 1.5, 1.95, 0),  # Bedroom
        (5.5, 1.5, 1.73, 2),  # Kitchen
        (1.0, 4.0, 1.13, 7),  # Bathroom
    ]
    boundary = np.array([[0,0],[7,0],[7,5],[0,5]], dtype=np.float32)
    constraints = generate_constraints_from_bubbles(bubbles, boundary, n_sample=20, seed=42)
    assert 1 <= len(constraints) <= 20
    assert all(isinstance(c, Constraint) for c in constraints)
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_constraints.py -v
```

**Step 3: Write implementation**

```python
# src/data/constraints.py
"""Constraint types, encoding, and procedural generation from floor plan bubbles."""
import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import combinations

NUM_CONSTRAINT_TYPES = 11
NUM_ROOM_TYPES = 9
CONSTRAINT_DIM = NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES + NUM_ROOM_TYPES + 1  # 30


class ConstraintType(IntEnum):
    MIN_AREA = 0
    MAX_AREA = 1
    MIN_COUNT = 2
    MAX_COUNT = 3
    MIN_DISTANCE = 4
    MAX_DISTANCE = 5
    MUST_EXIST = 6
    FORBIDDEN_ADJACENCY = 7
    REQUIRED_ADJACENCY = 8
    BOUNDARY_CONTACT = 9
    MIN_ROOM_RADIUS = 10


@dataclass
class Constraint:
    type: ConstraintType
    room_type_a: int  # -1 if unused
    room_type_b: int  # -1 if unused
    value: float      # 0.0 if unused


def encode_constraint(c: Constraint) -> np.ndarray:
    """Encode a constraint as a 30-dim vector."""
    vec = np.zeros(CONSTRAINT_DIM, dtype=np.float32)
    # Type one-hot (11 dims)
    vec[c.type] = 1.0
    # Room type A one-hot (9 dims, offset 11)
    if 0 <= c.room_type_a < NUM_ROOM_TYPES:
        vec[NUM_CONSTRAINT_TYPES + c.room_type_a] = 1.0
    # Room type B one-hot (9 dims, offset 20)
    if 0 <= c.room_type_b < NUM_ROOM_TYPES:
        vec[NUM_CONSTRAINT_TYPES + NUM_ROOM_TYPES + c.room_type_b] = 1.0
    # Value (1 dim, offset 29)
    vec[CONSTRAINT_DIM - 1] = c.value
    return vec


def generate_constraints_from_bubbles(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: np.ndarray,
    n_sample: int = 100,
    seed: Optional[int] = None,
    relax_factor: float = 0.1,
) -> List[Constraint]:
    """Procedurally generate constraints that the given bubble layout satisfies.

    Args:
        bubbles: list of (cx, cy, radius, room_type)
        boundary: (N, 2) boundary vertices
        n_sample: max number of constraints to sample
        relax_factor: fraction to relax continuous values by
    """
    rng = np.random.RandomState(seed)
    candidates = []

    if len(bubbles) == 0:
        return []

    types = [b[3] for b in bubbles]
    areas = [np.pi * b[2]**2 for b in bubbles]
    radii_vals = [b[2] for b in bubbles]
    unique_types = set(types)

    from collections import Counter
    type_counts = Counter(types)

    # MUST_EXIST for each present type
    for rt in unique_types:
        candidates.append(Constraint(ConstraintType.MUST_EXIST, rt, -1, 0.0))

    # MIN_COUNT / MAX_COUNT per type
    for rt, count in type_counts.items():
        candidates.append(Constraint(ConstraintType.MIN_COUNT, rt, -1, float(count)))
        candidates.append(Constraint(
            ConstraintType.MAX_COUNT, rt, -1,
            float(count + rng.randint(0, 3))
        ))

    # MIN_AREA / MAX_AREA per type
    for rt in unique_types:
        rt_areas = [a for b, a in zip(bubbles, areas) if b[3] == rt]
        min_a = min(rt_areas)
        max_a = max(rt_areas)
        candidates.append(Constraint(
            ConstraintType.MIN_AREA, rt, -1,
            min_a * (1.0 - relax_factor)
        ))
        candidates.append(Constraint(
            ConstraintType.MAX_AREA, rt, -1,
            max_a * (1.0 + relax_factor)
        ))

    # MIN_ROOM_RADIUS per type
    for rt in unique_types:
        rt_radii = [r for b, r in zip(bubbles, radii_vals) if b[3] == rt]
        candidates.append(Constraint(
            ConstraintType.MIN_ROOM_RADIUS, rt, -1,
            min(rt_radii) * (1.0 - relax_factor)
        ))

    # Pairwise distance constraints (sample pairs to avoid O(n^2) blowup)
    if len(bubbles) >= 2:
        pairs = list(combinations(range(len(bubbles)), 2))
        sampled_pairs = [pairs[i] for i in rng.choice(
            len(pairs), size=min(len(pairs), 50), replace=False
        )]
        for i, j in sampled_pairs:
            bi, bj = bubbles[i], bubbles[j]
            dist = np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2)
            # MAX_DISTANCE (satisfied: actual dist <= threshold)
            candidates.append(Constraint(
                ConstraintType.MAX_DISTANCE, bi[3], bj[3],
                dist * (1.0 + relax_factor)
            ))
            # Adjacency: if circles overlap or nearly touch
            touch_dist = bi[2] + bj[2]
            if dist <= touch_dist * 1.2:
                candidates.append(Constraint(
                    ConstraintType.REQUIRED_ADJACENCY, bi[3], bj[3], 0.0
                ))

    # FORBIDDEN_ADJACENCY: for type pairs that are NOT adjacent
    for rt_a in unique_types:
        for rt_b in unique_types:
            if rt_a >= rt_b:
                continue
            # Check if any pair of these types is adjacent
            has_adj = False
            for i, bi in enumerate(bubbles):
                if bi[3] != rt_a:
                    continue
                for j, bj in enumerate(bubbles):
                    if bj[3] != rt_b:
                        continue
                    dist = np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2)
                    if dist <= (bi[2] + bj[2]) * 1.2:
                        has_adj = True
                        break
                if has_adj:
                    break
            if not has_adj:
                candidates.append(Constraint(
                    ConstraintType.FORBIDDEN_ADJACENCY, rt_a, rt_b, 0.0
                ))

    # BOUNDARY_CONTACT: rooms whose circle touches boundary bbox
    bmin = boundary.min(axis=0)
    bmax = boundary.max(axis=0)
    for b in bubbles:
        cx, cy, r, rt = b
        if (cx - r <= bmin[0] + r * 0.5 or cx + r >= bmax[0] - r * 0.5 or
            cy - r <= bmin[1] + r * 0.5 or cy + r >= bmax[1] - r * 0.5):
            candidates.append(Constraint(
                ConstraintType.BOUNDARY_CONTACT, rt, -1, 0.0
            ))

    # Sample subset
    if len(candidates) <= n_sample:
        return candidates
    indices = rng.choice(len(candidates), size=n_sample, replace=False)
    return [candidates[i] for i in indices]
```

**Step 4: Run tests**

```bash
pytest tests/test_constraints.py -v
```

**Step 5: Commit**

```bash
git add src/data/constraints.py tests/test_constraints.py
git commit -m "feat: constraint system with procedural generation"
```

---

## Task 5: Dataset Class

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write failing test**

```python
# tests/test_dataset.py
import torch
from unittest.mock import patch
from src.data.dataset import BubbleDataset, collate_bubbles

def test_collate_shapes():
    """Test that collation produces correct padded tensor shapes."""
    # Simulate 2 samples with different sizes
    samples = [
        {
            "bubbles": torch.randn(5, 11),    # 5 rooms
            "bubble_mask": torch.ones(5),
            "boundary": torch.randn(8, 2),    # 8 vertices
            "boundary_mask": torch.ones(8),
            "constraints": torch.randn(10, 30), # 10 constraints
            "constraint_mask": torch.ones(10),
        },
        {
            "bubbles": torch.randn(3, 11),    # 3 rooms
            "bubble_mask": torch.ones(3),
            "boundary": torch.randn(6, 2),    # 6 vertices
            "boundary_mask": torch.ones(6),
            "constraints": torch.randn(7, 30), # 7 constraints
            "constraint_mask": torch.ones(7),
        },
    ]
    batch = collate_bubbles(samples)
    assert batch["bubbles"].shape == (2, 5, 11)       # padded to max rooms
    assert batch["bubble_mask"].shape == (2, 5)
    assert batch["boundary"].shape == (2, 8, 2)        # padded to max verts
    assert batch["constraints"].shape == (2, 10, 30)   # padded to max constraints
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_dataset.py -v
```

**Step 3: Write implementation**

```python
# src/data/dataset.py
"""PyTorch dataset for bubble diffusion training."""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional

from src.data.bubble_extractor import extract_bubbles, extract_boundary
from src.data.constraints import (
    generate_constraints_from_bubbles, encode_constraint, NUM_ROOM_TYPES
)


class BubbleDataset(Dataset):
    """Dataset of (boundary, constraints, bubbles) extracted from MSD."""

    def __init__(
        self,
        msd_path: str,
        split: str = "train",
        n_max: int = 200,
        n_boundary_max: int = 64,
        min_constraints: int = 10,
        max_constraints: int = 200,
        type_emb_dim: int = 8,
        augment: bool = True,
    ):
        self.n_max = n_max
        self.n_boundary_max = n_boundary_max
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.type_emb_dim = type_emb_dim
        self.augment = augment and (split == "train")

        msd = Path(msd_path)
        if split == "train":
            self.graph_dir = msd / "modified-swiss-dwellings-v1-train" / "graph_out"
            self.struct_dir = msd / "modified-swiss-dwellings-v1-train" / "struct_in"
        else:
            self.graph_dir = msd / "modified-swiss-dwellings-v1-test" / "graph_out"
            self.struct_dir = msd / "modified-swiss-dwellings-v1-test" / "struct_in"

        self.graph_files = sorted(self.graph_dir.glob("*.pickle"))
        self.struct_files = {f.stem: f for f in self.struct_dir.glob("*.npy")}

        # Type embedding: one-hot into type_emb_dim via fixed random projection
        # (will be replaced by learned embedding in the model, but we need a
        # target for diffusion)
        self.n_types = NUM_ROOM_TYPES + 1  # +1 for empty

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load graph
        with open(self.graph_files[idx], "rb") as f:
            G = pickle.load(f)

        # Extract bubbles
        bubbles = extract_bubbles(G)

        # Load struct_in for boundary
        stem = self.graph_files[idx].stem
        struct_file = self.struct_files.get(stem)
        if struct_file is not None:
            struct_in = np.load(struct_file)
            boundary = extract_boundary(struct_in)
        else:
            # Fallback: compute boundary from room geometries
            boundary = self._boundary_from_rooms(G)

        # Normalize to [0, 1]
        bubbles_arr, boundary_norm = self._normalize(bubbles, boundary)

        # Augmentation (random rotation/flip)
        if self.augment:
            bubbles_arr, boundary_norm = self._augment(bubbles_arr, boundary_norm)

        # Generate constraints
        bubble_list = [
            (b[0], b[1], b[2], int(b[3])) for b in bubbles_arr
        ]
        n_constraints = np.random.randint(self.min_constraints, self.max_constraints + 1)
        constraints = generate_constraints_from_bubbles(
            bubble_list, boundary_norm, n_sample=n_constraints
        )

        # Encode constraints
        if len(constraints) > 0:
            constraint_vecs = np.stack([encode_constraint(c) for c in constraints])
        else:
            constraint_vecs = np.zeros((1, 30), dtype=np.float32)

        # Build bubble tensor: [x, y, r, type_onehot(10)]
        n_rooms = len(bubbles_arr)
        bubble_tensor = np.zeros((n_rooms, 3 + self.n_types), dtype=np.float32)
        for i, b in enumerate(bubbles_arr):
            bubble_tensor[i, 0] = b[0]  # x
            bubble_tensor[i, 1] = b[1]  # y
            bubble_tensor[i, 2] = b[2]  # r
            bubble_tensor[i, 3 + int(b[3])] = 1.0  # type one-hot

        return {
            "bubbles": torch.from_numpy(bubble_tensor),
            "bubble_mask": torch.ones(n_rooms, dtype=torch.float32),
            "boundary": torch.from_numpy(boundary_norm),
            "boundary_mask": torch.ones(len(boundary_norm), dtype=torch.float32),
            "constraints": torch.from_numpy(constraint_vecs),
            "constraint_mask": torch.ones(len(constraint_vecs), dtype=torch.float32),
        }

    def _normalize(self, bubbles, boundary):
        """Normalize coordinates to [0, 1] based on boundary bounding box."""
        bmin = boundary.min(axis=0)
        bmax = boundary.max(axis=0)
        extent = bmax - bmin
        extent = np.maximum(extent, 1e-6)  # avoid division by zero

        boundary_norm = (boundary - bmin) / extent

        bubbles_norm = []
        for cx, cy, r, rt in bubbles:
            nx_ = (cx - bmin[0]) / extent[0]
            ny_ = (cy - bmin[1]) / extent[1]
            nr = r / max(extent)  # normalize radius by max extent
            bubbles_norm.append((nx_, ny_, nr, rt))

        return bubbles_norm, boundary_norm.astype(np.float32)

    def _augment(self, bubbles, boundary):
        """Random rotation (0/90/180/270) and horizontal flip."""
        rot = np.random.randint(4)
        flip = np.random.random() > 0.5

        def transform_point(x, y):
            for _ in range(rot):
                x, y = 1.0 - y, x
            if flip:
                x = 1.0 - x
            return x, y

        new_bubbles = []
        for cx, cy, r, rt in bubbles:
            nx, ny = transform_point(cx, cy)
            new_bubbles.append((nx, ny, r, rt))

        new_boundary = np.array([transform_point(x, y) for x, y in boundary],
                                dtype=np.float32)
        return new_bubbles, new_boundary

    def _boundary_from_rooms(self, G):
        """Fallback boundary from convex hull of all room geometries."""
        from shapely.ops import unary_union
        geoms = [d["geometry"] for _, d in G.nodes(data=True)
                 if "geometry" in d and d.get("room_type", -1) in range(10)]
        if not geoms:
            return np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
        union = unary_union(geoms)
        hull = union.convex_hull
        coords = np.array(hull.exterior.coords)[:-1]
        return coords.astype(np.float32)


def collate_bubbles(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate: pad variable-length tensors to batch max."""
    keys_and_dims = {
        "bubbles": 2,      # (N, 11)
        "bubble_mask": 1,   # (N,)
        "boundary": 2,      # (N, 2)
        "boundary_mask": 1,  # (N,)
        "constraints": 2,    # (N, 30)
        "constraint_mask": 1, # (N,)
    }

    result = {}
    for key, ndim in keys_and_dims.items():
        tensors = [sample[key] for sample in batch]
        max_len = max(t.shape[0] for t in tensors)

        padded = []
        for t in tensors:
            pad_size = max_len - t.shape[0]
            if pad_size > 0:
                if ndim == 1:
                    padded.append(torch.cat([t, torch.zeros(pad_size)]))
                else:
                    padded.append(torch.cat([
                        t, torch.zeros(pad_size, t.shape[1])
                    ]))
            else:
                padded.append(t)
        result[key] = torch.stack(padded)

    return result
```

**Step 4: Run tests**

```bash
pytest tests/test_dataset.py -v
```

**Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: BubbleDataset with collation and augmentation"
```

---

## Task 6: Diffusion Process (Noise Schedules + Forward/Reverse)

**Files:**
- Create: `src/model/diffusion.py`
- Create: `tests/test_diffusion.py`

**Step 1: Write failing test**

```python
# tests/test_diffusion.py
import torch
from src.model.diffusion import GaussianDiffusion

def test_forward_diffusion_shapes():
    diff = GaussianDiffusion(num_timesteps=1000, schedule="cosine")
    x0 = torch.randn(4, 50, 11)  # batch=4, 50 slots, 11 dims
    t = torch.randint(0, 1000, (4,))
    xt, noise = diff.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape

def test_noise_at_t0_is_clean():
    diff = GaussianDiffusion(num_timesteps=1000, schedule="cosine")
    x0 = torch.randn(2, 10, 11)
    t = torch.zeros(2, dtype=torch.long)
    xt, noise = diff.q_sample(x0, t)
    # At t=0, xt should be very close to x0
    assert (xt - x0).abs().max() < 0.05

def test_noise_at_tmax_is_noisy():
    diff = GaussianDiffusion(num_timesteps=1000, schedule="cosine")
    x0 = torch.ones(2, 10, 11)
    t = torch.full((2,), 999, dtype=torch.long)
    xt, noise = diff.q_sample(x0, t)
    # At t=T, xt should be mostly noise
    assert xt.std() > 0.5

def test_type_schedule_faster():
    diff = GaussianDiffusion(
        num_timesteps=1000, schedule="cosine", type_schedule_shift=0.3
    )
    # At t=300 (30% of 1000), type dims should be nearly clean
    # alpha_bar for type should be close to 1
    t = torch.tensor([300])
    alpha_bar_geom = diff.alphas_cumprod[t]
    alpha_bar_type = diff.alphas_cumprod_type[t]
    # Type should be cleaner (higher alpha_bar) than geometry at same t
    assert alpha_bar_type > alpha_bar_geom
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_diffusion.py -v
```

**Step 3: Write implementation**

```python
# src/model/diffusion.py
"""Gaussian diffusion process with separate type schedule."""
import torch
import torch.nn as nn
import numpy as np
import math


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


def shifted_cosine_schedule(timesteps: int, shift: float = 0.3, s: float = 0.008) -> torch.Tensor:
    """Faster cosine schedule: reaches ~0 noise by shift*T steps.

    We compress the cosine schedule into the first `shift` fraction of timesteps,
    then hold alpha_bar ~1 for the remaining steps.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    # Remap t so that t_effective goes 0->1 when t goes 0->shift
    t_effective = torch.clamp(t / shift, 0, 1)
    alphas_cumprod = torch.cos((t_effective + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    """Diffusion process with separate schedules for geometry and type dims."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        type_schedule_shift: float = 0.3,
        geom_dims: int = 3,   # x, y, r
        type_dims: int = 8,   # type embedding dims (will be set by config)
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.geom_dims = geom_dims
        self.type_dims = type_dims

        # Geometry schedule
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Type schedule (faster)
        betas_type = shifted_cosine_schedule(num_timesteps, shift=type_schedule_shift)
        alphas_type = 1.0 - betas_type
        alphas_cumprod_type = torch.cumprod(alphas_type, dim=0)

        # Register as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.register_buffer("alphas_cumprod_type", alphas_cumprod_type)
        self.register_buffer("sqrt_alphas_cumprod_type", torch.sqrt(alphas_cumprod_type))
        self.register_buffer("sqrt_one_minus_alphas_cumprod_type", torch.sqrt(1 - alphas_cumprod_type))

        # For DDPM reverse
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        ))
        self.register_buffer("posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
            (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """Forward diffusion: q(x_t | x_0).

        Args:
            x0: (B, N, D) where D = geom_dims + type_dims
            t: (B,) integer timesteps

        Returns:
            xt: noisy sample
            noise: the noise that was added
        """
        noise = torch.randn_like(x0)

        # Split into geometry and type
        x0_geom = x0[..., :self.geom_dims]
        x0_type = x0[..., self.geom_dims:]
        noise_geom = noise[..., :self.geom_dims]
        noise_type = noise[..., self.geom_dims:]

        # Broadcast schedule values: (B,) -> (B, 1, 1)
        sqrt_alpha_geom = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_geom = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_alpha_type = self.sqrt_alphas_cumprod_type[t][:, None, None]
        sqrt_one_minus_alpha_type = self.sqrt_one_minus_alphas_cumprod_type[t][:, None, None]

        xt_geom = sqrt_alpha_geom * x0_geom + sqrt_one_minus_alpha_geom * noise_geom
        xt_type = sqrt_alpha_type * x0_type + sqrt_one_minus_alpha_type * noise_type

        xt = torch.cat([xt_geom, xt_type], dim=-1)
        return xt, noise

    def predict_x0_from_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Recover x0 from xt and predicted noise."""
        xt_geom = xt[..., :self.geom_dims]
        xt_type = xt[..., self.geom_dims:]
        noise_geom = noise[..., :self.geom_dims]
        noise_type = noise[..., self.geom_dims:]

        sqrt_alpha_geom = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_geom = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_alpha_type = self.sqrt_alphas_cumprod_type[t][:, None, None]
        sqrt_one_minus_alpha_type = self.sqrt_one_minus_alphas_cumprod_type[t][:, None, None]

        x0_geom = (xt_geom - sqrt_one_minus_alpha_geom * noise_geom) / sqrt_alpha_geom
        x0_type = (xt_type - sqrt_one_minus_alpha_type * noise_type) / sqrt_alpha_type

        return torch.cat([x0_geom, x0_type], dim=-1)

    def p_sample(self, model_fn, xt: torch.Tensor, t: torch.Tensor, **model_kwargs):
        """Single DDPM reverse step."""
        pred_noise = model_fn(xt, t, **model_kwargs)

        # Posterior mean
        coef1 = self.posterior_mean_coef1[t][:, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None]
        x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)

        # Clamp x0 prediction
        x0_pred = torch.clamp(x0_pred, -3, 3)

        mean = coef1 * x0_pred + coef2 * xt

        # Add noise (except at t=0)
        noise = torch.randn_like(xt)
        log_var = self.posterior_log_variance[t][:, None, None]
        nonzero_mask = (t > 0).float()[:, None, None]

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, num_steps: int = 100,
                    cfg_scale: float = 0.0, **model_kwargs):
        """DDIM sampling loop.

        Args:
            model_fn: function(xt, t, **kwargs) -> predicted noise
            shape: (B, N, D) output shape
            num_steps: number of DDIM steps
            cfg_scale: classifier-free guidance scale (0 = no guidance)
        """
        device = self.betas.device
        b = shape[0]

        # Subsequence of timesteps for DDIM
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        xt = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)

            if cfg_scale > 0 and "null_kwargs" in model_kwargs:
                # Classifier-free guidance
                pred_cond = model_fn(xt, t, **{
                    k: v for k, v in model_kwargs.items() if k != "null_kwargs"
                })
                pred_uncond = model_fn(xt, t, **model_kwargs["null_kwargs"])
                pred_noise = (1 + cfg_scale) * pred_cond - cfg_scale * pred_uncond
            else:
                pred_noise = model_fn(xt, t, **model_kwargs)

            # DDIM update (eta=0 for deterministic)
            x0_pred = self.predict_x0_from_noise(xt, t, pred_noise)
            x0_pred = torch.clamp(x0_pred, -3, 3)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_tensor = torch.full((b,), t_next, device=device, dtype=torch.long)

                # Geometry dims
                alpha_t_g = self.alphas_cumprod[t_val]
                alpha_next_g = self.alphas_cumprod[t_next]
                x0_g = x0_pred[..., :self.geom_dims]
                eps_g = pred_noise[..., :self.geom_dims]
                xt_geom = (alpha_next_g.sqrt() * x0_g +
                          (1 - alpha_next_g).sqrt() * eps_g)

                # Type dims
                alpha_t_tp = self.alphas_cumprod_type[t_val]
                alpha_next_tp = self.alphas_cumprod_type[t_next]
                x0_tp = x0_pred[..., self.geom_dims:]
                eps_tp = pred_noise[..., self.geom_dims:]
                xt_type = (alpha_next_tp.sqrt() * x0_tp +
                          (1 - alpha_next_tp).sqrt() * eps_tp)

                xt = torch.cat([xt_geom, xt_type], dim=-1)
            else:
                xt = x0_pred

        return xt
```

**Step 4: Run tests**

```bash
pytest tests/test_diffusion.py -v
```

**Step 5: Commit**

```bash
git add src/model/diffusion.py tests/test_diffusion.py
git commit -m "feat: Gaussian diffusion with dual schedule for geometry and types"
```

---

## Task 7: Transformer Denoiser

**Files:**
- Create: `src/model/transformer.py`
- Create: `tests/test_transformer.py`

**Step 1: Write failing test**

```python
# tests/test_transformer.py
import torch
from src.model.transformer import BubbleDenoiser

def test_forward_shapes():
    model = BubbleDenoiser(
        slot_dim=13,  # 3 geom + 10 type onehot
        d_model=64,
        n_layers=2,
        n_heads=4,
        constraint_dim=30,
    )
    B, N, D = 2, 50, 13
    xt = torch.randn(B, N, D)
    t = torch.randint(0, 1000, (B,))
    bubble_mask = torch.ones(B, N)
    bubble_mask[1, 30:] = 0  # second sample has 30 rooms

    boundary = torch.randn(B, 8, 2)
    boundary_mask = torch.ones(B, 8)
    constraints = torch.randn(B, 20, 30)
    constraint_mask = torch.ones(B, 20)

    pred = model(xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask)
    assert pred.shape == (B, N, D)

def test_output_masked():
    """Output should be zero for masked (padded) slots."""
    model = BubbleDenoiser(slot_dim=13, d_model=64, n_layers=2, n_heads=4, constraint_dim=30)
    B, N, D = 2, 50, 13
    xt = torch.randn(B, N, D)
    t = torch.randint(0, 1000, (B,))
    bubble_mask = torch.ones(B, N)
    bubble_mask[0, 10:] = 0

    boundary = torch.randn(B, 8, 2)
    boundary_mask = torch.ones(B, 8)
    constraints = torch.randn(B, 5, 30)
    constraint_mask = torch.ones(B, 5)

    pred = model(xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask)
    # Masked positions should be zeroed
    assert pred[0, 10:].abs().max() < 1e-6
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_transformer.py -v
```

**Step 3: Write implementation**

```python
# src/model/transformer.py
"""Transformer denoiser for bubble diffusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNCrossAttentionBlock(nn.Module):
    """Transformer block with self-attention, two cross-attentions, and AdaLN-Zero."""

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # Cross-attention to boundary
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cross_attn_boundary = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # Cross-attention to constraints
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cross_attn_constraints = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        # FFN
        self.norm4 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

        # AdaLN modulation: 4 sub-layers x 3 params (shift, scale, gate) = 12
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 12 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, D)
        c: torch.Tensor,           # (B, D) timestep conditioning
        boundary: torch.Tensor,    # (B, Nb, D)
        boundary_mask: torch.Tensor,  # (B, Nb) 1=valid, 0=pad
        constraints: torch.Tensor,    # (B, Nc, D)
        constraint_mask: torch.Tensor, # (B, Nc)
        bubble_mask: torch.Tensor,     # (B, N)
    ) -> torch.Tensor:
        # Get modulation params
        params = self.adaLN(c).chunk(12, dim=-1)
        shift1, scale1, gate1 = params[0], params[1], params[2]
        shift2, scale2, gate2 = params[3], params[4], params[5]
        shift3, scale3, gate3 = params[6], params[7], params[8]
        shift4, scale4, gate4 = params[9], params[10], params[11]

        # Convert masks to attention format: True = ignore
        bubble_key_mask = (bubble_mask == 0)     # (B, N)
        boundary_key_mask = (boundary_mask == 0)  # (B, Nb)
        constraint_key_mask = (constraint_mask == 0)  # (B, Nc)

        # Self-attention
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.self_attn(h, h, h, key_padding_mask=bubble_key_mask)
        x = x + gate1.unsqueeze(1) * h

        # Cross-attention to boundary
        h = modulate(self.norm2(x), shift2, scale2)
        h, _ = self.cross_attn_boundary(h, boundary, boundary, key_padding_mask=boundary_key_mask)
        x = x + gate2.unsqueeze(1) * h

        # Cross-attention to constraints
        h = modulate(self.norm3(x), shift3, scale3)
        h, _ = self.cross_attn_constraints(h, constraints, constraints, key_padding_mask=constraint_key_mask)
        x = x + gate3.unsqueeze(1) * h

        # FFN
        h = modulate(self.norm4(x), shift4, scale4)
        h = self.ffn(h)
        x = x + gate4.unsqueeze(1) * h

        return x


class BubbleDenoiser(nn.Module):
    """Full denoiser: projects inputs, runs transformer blocks, projects output."""

    def __init__(
        self,
        slot_dim: int = 13,        # 3 + 10 (geom + type_onehot including empty)
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        constraint_dim: int = 30,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.d_model = d_model

        # Input projections
        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.boundary_proj = nn.Linear(2, d_model)
        self.constraint_proj = nn.Sequential(
            nn.Linear(constraint_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNCrossAttentionBlock(d_model, n_heads, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final norm + projection
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.output_proj = nn.Linear(d_model, slot_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        xt: torch.Tensor,              # (B, N, slot_dim)
        t: torch.Tensor,               # (B,)
        boundary: torch.Tensor,        # (B, Nb, 2)
        boundary_mask: torch.Tensor,   # (B, Nb)
        constraints: torch.Tensor,     # (B, Nc, constraint_dim)
        constraint_mask: torch.Tensor, # (B, Nc)
        bubble_mask: torch.Tensor,     # (B, N)
    ) -> torch.Tensor:
        # Project inputs
        x = self.slot_proj(xt)
        b_tokens = self.boundary_proj(boundary)
        c_tokens = self.constraint_proj(constraints)
        t_emb = self.time_emb(t)

        # Run transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, b_tokens, boundary_mask, c_tokens, constraint_mask, bubble_mask)

        # Final projection
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.output_proj(x)

        # Zero out padded positions
        x = x * bubble_mask.unsqueeze(-1)

        return x
```

**Step 4: Run tests**

```bash
pytest tests/test_transformer.py -v
```

**Step 5: Commit**

```bash
git add src/model/transformer.py tests/test_transformer.py
git commit -m "feat: transformer denoiser with AdaLN-Zero and dual cross-attention"
```

---

## Task 8: Training Loop

**Files:**
- Create: `src/training/trainer.py`
- Create: `scripts/train.py`

**Step 1: Write training loop**

```python
# src/training/trainer.py
"""Training loop for bubble diffusion."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time
import logging

from src.model.diffusion import GaussianDiffusion
from src.model.transformer import BubbleDenoiser
from src.data.dataset import BubbleDataset, collate_bubbles

logger = logging.getLogger(__name__)


class BubbleTrainer:
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self.global_step = 0

        dc = config["data"]
        mc = config["model"]
        tc = config["training"]
        diff_c = config["diffusion"]

        # Slot dim = 3 (geom) + num_room_types + 1 (empty)
        self.slot_dim = 3 + dc["num_room_types"] + 1  # 13

        # Model
        self.model = BubbleDenoiser(
            slot_dim=self.slot_dim,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            constraint_dim=dc["constraint_dim"],
            ffn_ratio=mc["ffn_ratio"],
            dropout=mc["dropout"],
        ).to(device)

        # Diffusion
        self.diffusion = GaussianDiffusion(
            num_timesteps=diff_c["num_timesteps"],
            schedule=diff_c["schedule"],
            type_schedule_shift=diff_c["type_schedule_shift"],
            geom_dims=3,
            type_dims=dc["num_room_types"] + 1,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tc["lr"], weight_decay=0.0
        )

        # LR scheduler
        self.warmup_steps = tc["warmup_steps"]
        self.max_steps = tc["max_steps"]

        # Training params
        self.cfg_drop_single = tc["cfg_drop_single"]
        self.cfg_drop_all = tc["cfg_drop_all"]
        self.cfg_drop_boundary = tc["cfg_drop_boundary"]
        self.type_loss_weight = tc["type_loss_weight"]
        self.grad_clip = tc["grad_clip"]

        # Dataset
        self.train_dataset = BubbleDataset(
            msd_path=dc["msd_path"],
            split="train",
            n_max=dc["n_max"],
            n_boundary_max=dc["n_boundary_max"],
            min_constraints=dc["min_constraints"],
            max_constraints=dc["max_constraints"],
            type_emb_dim=dc["num_room_types"] + 1,
            augment=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=tc["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_bubbles,
            pin_memory=True,
            drop_last=True,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

    def _get_lr(self) -> float:
        if self.global_step < self.warmup_steps:
            return self.config["training"]["lr"] * self.global_step / self.warmup_steps
        # Cosine decay
        progress = (self.global_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.config["training"]["lr"] * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    def _apply_cfg_dropout(self, constraints, constraint_mask, boundary, boundary_mask):
        """Apply classifier-free guidance dropout."""
        B = constraints.shape[0]

        # Drop all constraints for some samples
        drop_all = torch.rand(B, device=constraints.device) < self.cfg_drop_all
        constraint_mask = constraint_mask * (~drop_all).float().unsqueeze(-1)

        # Drop individual constraints
        if self.cfg_drop_single > 0:
            drop_single = torch.rand_like(constraint_mask) < self.cfg_drop_single
            constraint_mask = constraint_mask * (~drop_single).float()

        # Drop boundary for some samples
        drop_boundary = torch.rand(B, device=boundary.device) < self.cfg_drop_boundary
        boundary_mask = boundary_mask * (~drop_boundary).float().unsqueeze(-1)

        return constraints, constraint_mask, boundary, boundary_mask

    def train_step(self, batch: dict) -> dict:
        self.model.train()

        # Move to device
        bubbles = batch["bubbles"].to(self.device)          # (B, N, slot_dim)
        bubble_mask = batch["bubble_mask"].to(self.device)   # (B, N)
        boundary = batch["boundary"].to(self.device)         # (B, Nb, 2)
        boundary_mask = batch["boundary_mask"].to(self.device)
        constraints = batch["constraints"].to(self.device)   # (B, Nc, 30)
        constraint_mask = batch["constraint_mask"].to(self.device)

        B = bubbles.shape[0]

        # CFG dropout
        constraints, constraint_mask, boundary, boundary_mask = \
            self._apply_cfg_dropout(constraints, constraint_mask, boundary, boundary_mask)

        # Sample timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)

        # Forward diffusion
        xt, noise = self.diffusion.q_sample(bubbles, t)

        # Predict noise
        pred_noise = self.model(
            xt, t, boundary, boundary_mask, constraints, constraint_mask, bubble_mask
        )

        # Loss: MSE on noise, with masking
        # Geometry dims: masked for empty slots
        geom_error = (pred_noise[..., :3] - noise[..., :3]) ** 2
        geom_loss = (geom_error * bubble_mask.unsqueeze(-1)).sum() / bubble_mask.sum().clamp(min=1) / 3

        # Type dims: always active (model must predict empty type noise too)
        type_error = (pred_noise[..., 3:] - noise[..., 3:]) ** 2
        n_type_dims = self.slot_dim - 3
        type_loss = type_error.sum() / (B * pred_noise.shape[1] * n_type_dims)

        loss = geom_loss + self.type_loss_weight * type_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # LR schedule
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()
        self.global_step += 1

        return {
            "loss": loss.item(),
            "geom_loss": geom_loss.item(),
            "type_loss": type_loss.item(),
            "lr": lr,
        }

    def train(self):
        logger.info(f"Starting training for {self.max_steps} steps")
        tc = self.config["training"]
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)

        data_iter = iter(self.train_loader)

        for step in range(self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            if step % 100 == 0:
                logger.info(
                    f"Step {step}: loss={metrics['loss']:.4f} "
                    f"geom={metrics['geom_loss']:.4f} "
                    f"type={metrics['type_loss']:.4f} "
                    f"lr={metrics['lr']:.6f}"
                )

            if step > 0 and step % tc["save_every"] == 0:
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.global_step,
                    "config": self.config,
                }
                torch.save(ckpt, save_dir / f"checkpoint_{step}.pt")
                logger.info(f"Saved checkpoint at step {step}")

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]
```

**Step 2: Write train script**

```python
# scripts/train.py
"""Launch training."""
import yaml
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.trainer import BubbleTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = BubbleTrainer(config, device="cuda")
    trainer.train()

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/training/trainer.py scripts/train.py
git commit -m "feat: training loop with CFG dropout and dual loss"
```

---

## Task 9: Inference and Sampling

**Files:**
- Create: `src/inference/sampler.py`
- Create: `tests/test_sampler.py`

**Step 1: Write failing test**

```python
# tests/test_sampler.py
import torch
from src.inference.sampler import BubbleSampler
from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion

def test_sample_shapes():
    model = BubbleDenoiser(slot_dim=13, d_model=64, n_layers=2, n_heads=4, constraint_dim=30)
    diffusion = GaussianDiffusion(num_timesteps=100, geom_dims=3, type_dims=10)
    sampler = BubbleSampler(model, diffusion, num_room_types=9)

    boundary = torch.randn(1, 8, 2)
    boundary_mask = torch.ones(1, 8)
    constraints = torch.randn(1, 15, 30)
    constraint_mask = torch.ones(1, 15)

    bubbles = sampler.sample(
        boundary, boundary_mask, constraints, constraint_mask,
        n_slots=50, num_steps=10, cfg_scale=0.0,
    )
    # bubbles: list of (cx, cy, r, room_type) after filtering empty
    assert isinstance(bubbles, list)
    assert all(len(b) == 4 for b in bubbles)
    assert all(isinstance(b[3], int) for b in bubbles)
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_sampler.py -v
```

**Step 3: Write implementation**

```python
# src/inference/sampler.py
"""Sampling / inference for bubble diffusion."""
import torch
import numpy as np
from typing import List, Tuple, Optional

from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion


class BubbleSampler:
    def __init__(
        self,
        model: BubbleDenoiser,
        diffusion: GaussianDiffusion,
        num_room_types: int = 9,
    ):
        self.model = model
        self.diffusion = diffusion
        self.num_room_types = num_room_types
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        boundary: torch.Tensor,          # (1, Nb, 2)
        boundary_mask: torch.Tensor,      # (1, Nb)
        constraints: torch.Tensor,        # (1, Nc, 30)
        constraint_mask: torch.Tensor,    # (1, Nc)
        n_slots: int = 200,
        num_steps: int = 100,
        cfg_scale: float = 3.0,
        energy_fns: Optional[List] = None,
        energy_lambda: float = 0.01,
    ) -> List[Tuple[float, float, float, int]]:
        """Generate bubbles via DDIM sampling.

        Returns list of (cx, cy, radius, room_type) for non-empty slots.
        """
        self.model.eval()
        B = 1
        slot_dim = self.model.slot_dim

        boundary = boundary.to(self.device)
        boundary_mask = boundary_mask.to(self.device)
        constraints = constraints.to(self.device)
        constraint_mask = constraint_mask.to(self.device)
        bubble_mask = torch.ones(B, n_slots, device=self.device)

        def model_fn(xt, t, **kwargs):
            bm = kwargs.get("boundary_mask", boundary_mask)
            cm = kwargs.get("constraint_mask", constraint_mask)
            return self.model(
                xt, t,
                kwargs.get("boundary", boundary), bm,
                kwargs.get("constraints", constraints), cm,
                bubble_mask,
            )

        # Prepare null kwargs for CFG
        null_constraint_mask = torch.zeros_like(constraint_mask)
        null_kwargs = {
            "boundary": boundary,
            "boundary_mask": boundary_mask,
            "constraints": constraints,
            "constraint_mask": null_constraint_mask,
        }

        # DDIM sampling
        shape = (B, n_slots, slot_dim)

        if cfg_scale > 0:
            # Manual DDIM loop with CFG
            xt = torch.randn(shape, device=self.device)
            step_size = self.diffusion.num_timesteps // num_steps
            timesteps = list(range(0, self.diffusion.num_timesteps, step_size))[::-1]

            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=self.device, dtype=torch.long)

                # Conditional prediction
                pred_cond = model_fn(xt, t)
                # Unconditional prediction
                pred_uncond = model_fn(
                    xt, t,
                    constraint_mask=null_constraint_mask,
                )
                # CFG
                pred_noise = (1 + cfg_scale) * pred_cond - cfg_scale * pred_uncond

                # DDIM update
                x0_pred = self.diffusion.predict_x0_from_noise(xt, t, pred_noise)
                x0_pred = torch.clamp(x0_pred, -3, 3)

                # Optional energy guidance
                if energy_fns and energy_lambda > 0:
                    x0_req_grad = x0_pred.detach().requires_grad_(True)
                    total_energy = sum(fn(x0_req_grad) for fn in energy_fns)
                    total_energy.backward()
                    x0_pred = x0_pred - energy_lambda * x0_req_grad.grad

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    # Geometry
                    ag = self.diffusion.alphas_cumprod[t_val]
                    ang = self.diffusion.alphas_cumprod[t_next]
                    x0_g = x0_pred[..., :3]
                    eps_g = pred_noise[..., :3]
                    xt_g = ang.sqrt() * x0_g + (1 - ang).sqrt() * eps_g
                    # Type
                    at = self.diffusion.alphas_cumprod_type[t_val]
                    ant = self.diffusion.alphas_cumprod_type[t_next]
                    x0_t = x0_pred[..., 3:]
                    eps_t = pred_noise[..., 3:]
                    xt_t = ant.sqrt() * x0_t + (1 - ant).sqrt() * eps_t
                    xt = torch.cat([xt_g, xt_t], dim=-1)
                else:
                    xt = x0_pred
        else:
            xt = self.diffusion.ddim_sample(model_fn, shape, num_steps=num_steps)

        # Decode: snap types, filter empty
        result = xt[0].cpu().numpy()  # (N, slot_dim)
        bubbles = []

        for slot in result:
            cx, cy, r = slot[0], slot[1], slot[2]
            type_logits = slot[3:]  # (num_room_types + 1,)
            type_idx = int(np.argmax(type_logits))

            # Last index = empty type
            if type_idx == self.num_room_types:
                continue

            # Clamp to valid range
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            r = np.clip(r, 0.01, 0.5)

            bubbles.append((float(cx), float(cy), float(r), type_idx))

        return bubbles
```

**Step 4: Run tests**

```bash
pytest tests/test_sampler.py -v
```

**Step 5: Commit**

```bash
git add src/inference/sampler.py tests/test_sampler.py
git commit -m "feat: DDIM sampler with CFG and optional energy guidance"
```

---

## Task 10: Evaluation and Visualization

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/visualize.py`
- Create: `scripts/evaluate.py`

**Step 1: Write metrics**

```python
# src/evaluation/metrics.py
"""Evaluation metrics for generated bubble layouts."""
import numpy as np
from typing import List, Tuple
from src.data.constraints import Constraint, ConstraintType


def constraint_satisfaction_rate(
    bubbles: List[Tuple[float, float, float, int]],
    constraints: List[Constraint],
    adjacency_threshold: float = 0.15,
) -> float:
    """Fraction of constraints satisfied by the bubble layout."""
    if not constraints:
        return 1.0

    satisfied = 0
    for c in constraints:
        if _check_constraint(bubbles, c, adjacency_threshold):
            satisfied += 1
    return satisfied / len(constraints)


def _check_constraint(bubbles, c, adj_thresh):
    types = [b[3] for b in bubbles]
    areas = [np.pi * b[2]**2 for b in bubbles]
    from collections import Counter
    counts = Counter(types)

    if c.type == ConstraintType.MUST_EXIST:
        return c.room_type_a in types
    elif c.type == ConstraintType.MIN_COUNT:
        return counts.get(c.room_type_a, 0) >= c.value
    elif c.type == ConstraintType.MAX_COUNT:
        return counts.get(c.room_type_a, 0) <= c.value
    elif c.type == ConstraintType.MIN_AREA:
        rt_areas = [a for b, a in zip(bubbles, areas) if b[3] == c.room_type_a]
        return all(a >= c.value for a in rt_areas) if rt_areas else False
    elif c.type == ConstraintType.MAX_AREA:
        rt_areas = [a for b, a in zip(bubbles, areas) if b[3] == c.room_type_a]
        return all(a <= c.value for a in rt_areas) if rt_areas else True
    elif c.type == ConstraintType.MIN_DISTANCE:
        for bi in bubbles:
            if bi[3] != c.room_type_a:
                continue
            for bj in bubbles:
                if bj[3] != c.room_type_b:
                    continue
                d = np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2)
                if d < c.value:
                    return False
        return True
    elif c.type == ConstraintType.MAX_DISTANCE:
        for bi in bubbles:
            if bi[3] != c.room_type_a:
                continue
            for bj in bubbles:
                if bj[3] != c.room_type_b:
                    continue
                if np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2) <= c.value:
                    return True
        return not (c.room_type_a in types and c.room_type_b in types)
    elif c.type == ConstraintType.REQUIRED_ADJACENCY:
        for bi in bubbles:
            if bi[3] != c.room_type_a:
                continue
            for bj in bubbles:
                if bj[3] != c.room_type_b:
                    continue
                d = np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2)
                if d <= bi[2] + bj[2] + adj_thresh:
                    return True
        return False
    elif c.type == ConstraintType.FORBIDDEN_ADJACENCY:
        for bi in bubbles:
            if bi[3] != c.room_type_a:
                continue
            for bj in bubbles:
                if bj[3] != c.room_type_b:
                    continue
                d = np.sqrt((bi[0]-bj[0])**2 + (bi[1]-bj[1])**2)
                if d <= bi[2] + bj[2] + adj_thresh:
                    return False
        return True
    elif c.type == ConstraintType.BOUNDARY_CONTACT:
        for b in bubbles:
            if b[3] == c.room_type_a:
                if (b[0] - b[2] <= 0.05 or b[0] + b[2] >= 0.95 or
                    b[1] - b[2] <= 0.05 or b[1] + b[2] >= 0.95):
                    return True
        return c.room_type_a not in types
    elif c.type == ConstraintType.MIN_ROOM_RADIUS:
        for b in bubbles:
            if b[3] == c.room_type_a and b[2] < c.value:
                return False
        return True
    return True


def boundary_coverage(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: np.ndarray,
) -> float:
    """Ratio of circle area inside boundary to boundary area."""
    from shapely.geometry import Polygon, Point
    if len(bubbles) == 0:
        return 0.0
    poly = Polygon(boundary)
    if poly.area == 0:
        return 0.0
    total = 0.0
    for cx, cy, r, _ in bubbles:
        circle = Point(cx, cy).buffer(r)
        total += circle.intersection(poly).area
    return min(total / poly.area, 1.0)
```

**Step 2: Write visualization**

```python
# src/evaluation/visualize.py
"""Visualization of bubble layouts."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional

ROOM_COLORS = {
    0: "#4E79A7",  # Bedroom - blue
    1: "#F28E2B",  # Living room - orange
    2: "#E15759",  # Kitchen - red
    3: "#76B7B2",  # Dining - teal
    4: "#B0B0B0",  # Corridor - gray
    5: "#EDC948",  # Stairs - yellow
    6: "#BAB0AC",  # Storeroom - tan
    7: "#59A14F",  # Bathroom - green
    8: "#AF7AA1",  # Balcony - purple
}

ROOM_NAMES = [
    "Bedroom", "Living", "Kitchen", "Dining", "Corridor",
    "Stairs", "Storage", "Bathroom", "Balcony"
]


def plot_bubbles(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
):
    """Plot bubble diagram overlaid on boundary."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw boundary
    if boundary is not None:
        closed = np.vstack([boundary, boundary[0]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=2)
        ax.fill(closed[:, 0], closed[:, 1], alpha=0.05, color="gray")

    # Draw bubbles
    for cx, cy, r, rt in bubbles:
        color = ROOM_COLORS.get(rt, "#333333")
        circle = plt.Circle((cx, cy), r, color=color, alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(
            ROOM_NAMES[rt] if rt < len(ROOM_NAMES) else f"T{rt}",
            (cx, cy), ha="center", va="center", fontsize=7,
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title)

    # Legend
    legend_handles = []
    present_types = sorted(set(b[3] for b in bubbles))
    for rt in present_types:
        legend_handles.append(
            patches.Patch(color=ROOM_COLORS.get(rt, "#333"), alpha=0.5,
                         label=ROOM_NAMES[rt] if rt < len(ROOM_NAMES) else f"T{rt}")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
```

**Step 3: Write eval script**

```python
# scripts/evaluate.py
"""Run evaluation on trained model."""
import yaml
import torch
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion
from src.inference.sampler import BubbleSampler
from src.data.dataset import BubbleDataset, collate_bubbles
from src.evaluation.metrics import constraint_satisfaction_rate, boundary_coverage
from src.evaluation.visualize import plot_bubbles
from src.data.constraints import generate_constraints_from_bubbles, Constraint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/latest.pt"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dc = config["data"]
    mc = config["model"]
    slot_dim = 3 + dc["num_room_types"] + 1

    # Load model
    model = BubbleDenoiser(
        slot_dim=slot_dim,
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        constraint_dim=dc["constraint_dim"],
    ).cuda()

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])

    diffusion = GaussianDiffusion(
        num_timesteps=config["diffusion"]["num_timesteps"],
        geom_dims=3,
        type_dims=dc["num_room_types"] + 1,
    ).cuda()

    sampler = BubbleSampler(model, diffusion)

    # Load validation data for constraint generation
    dataset = BubbleDataset(dc["msd_path"], split="train", augment=False)

    n_eval = config["eval"]["num_samples"]
    csr_scores = []
    coverage_scores = []

    output_dir = Path("eval_outputs")
    output_dir.mkdir(exist_ok=True)

    for i in range(min(n_eval, len(dataset))):
        sample = dataset[i]

        boundary = sample["boundary"].unsqueeze(0)
        boundary_mask = sample["boundary_mask"].unsqueeze(0)
        constraints_t = sample["constraints"].unsqueeze(0)
        constraint_mask = sample["constraint_mask"].unsqueeze(0)

        bubbles = sampler.sample(
            boundary, boundary_mask, constraints_t, constraint_mask,
            n_slots=dc["n_max"],
            num_steps=config["inference"]["num_steps"],
            cfg_scale=config["inference"]["cfg_scale"],
        )

        # Evaluate
        # Note: we'd need the original Constraint objects for proper evaluation
        # For now, just check coverage and visualize
        bnd = sample["boundary"].numpy()
        cov = boundary_coverage(bubbles, bnd)
        coverage_scores.append(cov)

        if i < 20:  # Visualize first 20
            plot_bubbles(bubbles, bnd, title=f"Sample {i} (coverage={cov:.2f})",
                        save_path=str(output_dir / f"sample_{i:03d}.png"))

        if i % 50 == 0:
            logger.info(f"Evaluated {i}/{n_eval}")

    logger.info(f"\nResults over {len(coverage_scores)} samples:")
    logger.info(f"  Boundary coverage: {sum(coverage_scores)/len(coverage_scores):.3f}")


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add src/evaluation/ scripts/evaluate.py
git commit -m "feat: evaluation metrics, visualization, and eval script"
```

---

## Task 11: End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write end-to-end test**

```python
# tests/test_e2e.py
"""End-to-end smoke test: create fake data, train 10 steps, sample."""
import torch
import numpy as np
from src.model.transformer import BubbleDenoiser
from src.model.diffusion import GaussianDiffusion
from src.training.trainer import BubbleTrainer
from src.inference.sampler import BubbleSampler
from src.data.dataset import collate_bubbles

def test_train_and_sample():
    """Minimal e2e: build model, run a few train steps on fake data, then sample."""
    slot_dim = 13  # 3 + 10

    model = BubbleDenoiser(
        slot_dim=slot_dim, d_model=64, n_layers=2, n_heads=4, constraint_dim=30,
    ).cuda()

    diffusion = GaussianDiffusion(
        num_timesteps=100, geom_dims=3, type_dims=10,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch
    B, N = 4, 20
    batch = {
        "bubbles": torch.randn(B, N, slot_dim).cuda(),
        "bubble_mask": torch.ones(B, N).cuda(),
        "boundary": torch.randn(B, 8, 2).cuda(),
        "boundary_mask": torch.ones(B, 8).cuda(),
        "constraints": torch.randn(B, 10, 30).cuda(),
        "constraint_mask": torch.ones(B, 10).cuda(),
    }

    # Train 10 steps
    model.train()
    for _ in range(10):
        t = torch.randint(0, 100, (B,)).cuda()
        xt, noise = diffusion.q_sample(batch["bubbles"], t)
        pred = model(xt, t, batch["boundary"], batch["boundary_mask"],
                     batch["constraints"], batch["constraint_mask"],
                     batch["bubble_mask"])
        loss = ((pred - noise) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 10.0  # sanity: loss should be finite

    # Sample
    sampler = BubbleSampler(model, diffusion, num_room_types=9)
    bubbles = sampler.sample(
        batch["boundary"][:1], batch["boundary_mask"][:1],
        batch["constraints"][:1], batch["constraint_mask"][:1],
        n_slots=20, num_steps=10, cfg_scale=0.0,
    )
    assert isinstance(bubbles, list)
```

**Step 2: Run test**

```bash
pytest tests/test_e2e.py -v
```

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end smoke test for train + sample"
```

---

## Summary of Tasks

| Task | What | Key files |
|------|------|-----------|
| 1 | Project setup, venv, deps | pyproject.toml, configs/default.yaml |
| 2 | Explore MSD, determine N_max | scripts/explore_msd.py |
| 3 | Bubble extraction from MSD | src/data/bubble_extractor.py |
| 4 | Constraint system | src/data/constraints.py |
| 5 | Dataset class + collation | src/data/dataset.py |
| 6 | Diffusion process (dual schedule) | src/model/diffusion.py |
| 7 | Transformer denoiser | src/model/transformer.py |
| 8 | Training loop | src/training/trainer.py |
| 9 | Inference / sampling | src/inference/sampler.py |
| 10 | Evaluation + visualization | src/evaluation/ |
| 11 | End-to-end smoke test | tests/test_e2e.py |
