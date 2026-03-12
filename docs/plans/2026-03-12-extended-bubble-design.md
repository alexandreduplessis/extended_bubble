# Extended Bubble Graph Generation — Design Document

**Date**: 2026-03-12
**Goal**: Generate approximate room layouts (center, radius, type) from a building boundary and a variable set of architectural constraints, as the first step of a two-stage floor plan generation pipeline.

---

## 1. Problem Statement

Given:
- A floor boundary polygon (outer wall outline of one floor)
- A variable-length set of architectural constraints (100–500), both continuous (area bounds, distances) and discrete (room existence, adjacency rules)

Produce:
- A set of "extended bubbles": (center_x, center_y, radius, room_type) for each room
- This is an approximate layout — exact room shapes and architectural details (doors, windows) are deferred to step 2

The output serves as a structured condition map for a downstream pixel-level generative model (e.g., the diffusion model from Zhang & Zhang 2025).

---

## 2. Why This Approach

### Two-stage rationale
Generating detailed floor plans directly from high-level constraints is extremely hard — the constraint space is vast and the output is high-dimensional (pixel images). By first generating a coarse bubble layout, we:
1. Decompose the problem into tractable sub-problems
2. Get an intermediate representation that's easy to evaluate and steer
3. Enable constraint satisfaction at a level where it's geometrically meaningful

### Why diffusion
- The output is an unordered set of variable size — diffusion is naturally permutation-equivariant
- Classifier-free guidance and energy-based guidance allow steering with new constraints at inference without retraining
- Strong precedent in molecular generation (EDM, GeoDiff) for similar set-generation problems

### Why not autoregressive
- Global constraints (e.g., "floor needs ≥ 2 compartments") require seeing the whole layout — autoregressive models commit early and may need expensive backtracking
- Ordering rooms is arbitrary and adds unnecessary complexity

---

## 3. Data Pipeline

### Source
**Modified Swiss Dwellings (MSD)** dataset from Kaggle:
- 5,372 floor plans, 163K+ rooms, 18.9K apartments
- Polygon geometries in WKT format with semantic labels
- 9 room types used (excluding Background and Structure):
  - 0: Bedroom, 1: Living room, 2: Kitchen, 3: Dining, 4: Corridor, 5: Stairs, 6: Storeroom, 7: Bathroom, 8: Balcony
- CC BY-SA 4.0 license
- Download: https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings

### Per-floor-plan extraction
1. **Boundary polygon**: Extract outer wall polygon, normalize vertices to [0, 1] bounding box
2. **Room bubbles**: For each room polygon:
   - center = centroid of polygon
   - radius = sqrt(area / pi) (area-preserving circle)
   - room_type from MSD label (0–8)
3. **Constraint generation**: Procedurally generate constraints (see Section 4)

### Data augmentation
- 4 rotations (0, 90, 180, 270) x 2 flips (none, horizontal) = 8x geometric augmentation
- Transform boundary vertices and bubble centers accordingly
- Sample different constraint subsets each epoch -> effectively unlimited training pairs

### Training pairs
Each training example is: (boundary_vertices, constraint_set, bubble_list)
- boundary_vertices: ordered list of (x, y) in [0, 1]
- constraint_set: variable-length list of constraint vectors
- bubble_list: list of (x, y, r, type) in [0, 1] normalized space

---

## 4. Constraint System

### Constraint type vocabulary

| Type | Params | Example |
|------|--------|---------|
| MIN_AREA | room_type, value | Bedroom area >= 9 m^2 |
| MAX_AREA | room_type, value | Bathroom area <= 12 m^2 |
| MIN_COUNT | room_type, value | >= 2 bedrooms |
| MAX_COUNT | room_type, value | <= 1 kitchen |
| MIN_DISTANCE | type_a, type_b, value | Living room <-> WC >= 3 m |
| MAX_DISTANCE | type_a, type_b, value | Bedroom <-> Bathroom <= 8 m |
| MUST_EXIST | room_type | Must have a kitchen |
| FORBIDDEN_ADJACENCY | type_a, type_b | No WC adjacent to kitchen |
| REQUIRED_ADJACENCY | type_a, type_b | Bathroom adjacent to bedroom |
| BOUNDARY_CONTACT | room_type | Living room must touch boundary |
| MIN_ROOM_RADIUS | room_type, value | Living room radius >= 2 m |

### Procedural constraint generation from MSD
For each floor plan:
1. Evaluate all constraint templates against the ground truth room polygons
2. Identify which constraints are satisfied
3. Sample a random subset of size k ~ Uniform(10, 200)
4. For distance/area constraints: extract actual values from the floor plan, optionally relax by a small margin

### Constraint encoding
Each constraint is encoded as a fixed-size vector:
```
[type_onehot (11 dims), room_type_a_onehot (9 dims), room_type_b_onehot (9 dims), value (1 dim)]
-> 30 dims total
```
This is passed through a small MLP (2 layers, ReLU) to produce a d_model-dimensional token.

### Adjacency definition for training
Two rooms are considered adjacent if: distance(centroid_a, centroid_b) <= radius_a + radius_b + epsilon.
This is a soft geometric proxy computable from the bubble representation.

---

## 5. Model Architecture

### Overview
Slot-based transformer denoiser operating on N_max bubble slots, conditioned on boundary and constraints via cross-attention.

### Bubble slot representation
- N_max slots (to be determined from MSD, estimated ~150–200)
- Each slot: [x, y, r, type_emb] = 11 continuous dims
  - (x, y, r): 3 continuous dims, normalized to [0, 1]
  - type_emb: 9 room types + 1 "empty" type, embedded via learned table into 8 continuous dims
- Empty slots use the "empty" type; loss on (x, y, r) is masked for empty slots

### Denoiser architecture
```
Input: noisy slots Z_t (N_max x 11), timestep t, boundary B, constraints C

1. Linear projection: Z_t -> (N_max x d_model)
2. Boundary encoding: each vertex (x, y) -> linear -> (N_boundary x d_model)
3. Constraint encoding: each constraint (30 dims) -> MLP -> (N_constraints x d_model)
4. Timestep: sinusoidal embedding -> MLP -> used for AdaLN modulation

5. For each of L transformer layers:
   a. AdaLN-modulated self-attention over bubble slots
   b. Cross-attention: bubbles (Q) x boundary (K, V)
   c. Cross-attention: bubbles (Q) x constraints (K, V)
   d. AdaLN-modulated FFN

6. Linear projection: (N_max x d_model) -> (N_max x 11) = predicted noise
```

### Noise schedule
- (x, y, r) dims: standard cosine beta schedule
- type_emb dims: faster cosine schedule (shifted so types are ~clean by t = 0.3T)
  - This ensures room types crystallize early in the diffusion process

### Classifier-free guidance
- Training: drop each constraint independently with p=0.1, drop all constraints with p=0.05, drop boundary with p=0.02
- Inference: guidance scale w ~ 2–4 (to be tuned)
  - prediction = (1 + w) * conditional_prediction - w * unconditional_prediction

### Model sizing (for RTX A5000, 24GB)
- d_model = 256
- L = 8 transformer layers
- 8 attention heads
- N_max ~ 200 slots
- ~50 boundary tokens, ~200 constraint tokens
- ~15M parameters
- Batch size: 32–64
- Fits comfortably in 24GB VRAM

---

## 6. Training

### Loss function
Standard diffusion MSE on predicted noise with modifications:
- **Empty slot masking**: (x, y, r) loss is zero on empty slots; type loss is always active (model must learn to predict "empty")
- **Optional type weighting**: Upweight type dim loss since type accuracy is more important than exact geometry

### Optimizer
- Adam, lr = 1e-4
- Linear warmup: 1K steps
- Cosine decay schedule

### Training budget
- ~200K steps, batch size 32–64
- Estimated ~1–2 days on the A5000

### Validation
- Hold out ~10% of floor plans by building (no building overlap, following MSD's original split)
- Monitor: validation loss, constraint satisfaction rate on generated samples

---

## 7. Inference

1. Sample Z_T ~ N(0, I) for N_max slots
2. Denoise using DDIM (50–100 steps for speed)
3. Apply classifier-free guidance at each step
4. Optionally apply energy-based guidance for hard constraints:
   - Define E(Z) as differentiable penalty per constraint
   - E.g., E_min_area = ReLU(min_value - pi * r^2)
   - E.g., E_max_distance = ReLU(dist(a, b) - max_value)
   - Update: Z_t <- Z_t - lambda * grad(E(Z_t))
5. At t=0: snap type_emb to nearest room type embedding, discard "empty" slots
6. Output: list of (center_x, center_y, radius, room_type)

---

## 8. Evaluation Metrics

- **Constraint satisfaction rate**: % of input constraints satisfied by generated output
- **Boundary coverage**: Ratio of (union of circle areas inside boundary) / (boundary area)
- **Type distribution**: KL divergence between generated and MSD room type frequencies
- **Visual inspection**: Plot bubble diagrams overlaid on boundary polygon
- **Distributional metrics**: Compare distributions of room count, area distribution, and inter-room distances between generated and real layouts

---

## 9. Key Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| Diffusion over autoregressive | Global constraints need full-layout reasoning; AR commits early |
| Slot-based with N_max | Handles variable output size; empty-slot mechanism well-proven in molecular generation |
| Constraints as cross-attention tokens | Variable-length input handled naturally; each constraint contributes via attention |
| Classifier-free + energy guidance | CFG handles bulk of constraint satisfaction; energy guidance as safety net for hard/novel constraints |
| Faster type schedule | Room types should crystallize early; geometry refinement continues |
| Continuous type relaxation | Simplest approach; snap to discrete types at end |
| Procedural constraint generation | Creates diverse training data from limited floor plans; constraint dropout adds further variety |
| No doors/windows in v1 | Keeps scope clean; these are detail-level and better handled in step 2 |
| No equivariance | Boundary pins the coordinate frame; normalize to [0,1] instead |
| Separate cross-attention for boundary vs constraints | Semantically different inputs; cleaner separation |

---

## 10. Project Structure

```
extended_bubble/
  docs/plans/          # This design doc and future plans
  src/
    data/              # MSD loading, bubble extraction, constraint generation
    model/             # Transformer denoiser, diffusion process
    training/          # Training loop, logging
    inference/         # Sampling, energy guidance
    evaluation/        # Metrics, visualization
  configs/             # Hyperparameter configs
  scripts/             # Download data, run training, run eval
```

Data and environments stored at: /Data/amine.chraibi/

---

## 11. Open Questions / Future Work

- Exact N_max to be determined from MSD room count statistics
- Type loss weighting to be tuned empirically
- Energy guidance strength (lambda) and which constraints benefit most from it
- Whether to add doors/windows in a future version
- Integration with step 2 (pixel-level generation model)
