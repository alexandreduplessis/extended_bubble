"""Differentiable energy functions for guided sampling.

Each energy function takes x0_pred (B, N, slot_dim) and returns a scalar energy
to minimize. The sampler computes gradients of these energies w.r.t. x0 to
nudge the denoised prediction toward constraint satisfaction.

All geometric energies are weighted by p_active (probability of being non-empty)
so that empty slots don't contribute to boundary/overlap penalties.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable


NUM_GEOM_DIMS = 3  # cx, cy, r


def _get_soft_types(x0: torch.Tensor, num_room_types: int = 9):
    """Extract soft type probabilities from x0.

    Returns:
        p_active: (N,) probability of being a real room (not empty)
        room_probs: (N, num_room_types) per-type probabilities
    """
    type_logits = x0[0, :, NUM_GEOM_DIMS:]  # (N, num_types+1)
    type_probs = F.softmax(type_logits * 5.0, dim=-1)
    p_active = 1.0 - type_probs[:, -1]  # last dim = empty
    room_probs = type_probs[:, :num_room_types]
    return p_active, room_probs


def make_boundary_energy(
    boundary: torch.Tensor,
    boundary_mask: torch.Tensor,
    num_room_types: int = 9,
    margin: float = 0.0,
) -> Callable:
    """Penalize bubble centers outside the boundary polygon.

    Weighted by p_active so empty slots don't get pushed inside.
    """
    valid = boundary_mask[0] > 0.5
    bnd = boundary[0][valid]  # (M, 2)

    if len(bnd) < 3:
        return lambda x0: torch.tensor(0.0, device=x0.device)

    # Precompute edge normals (inward-pointing)
    edges = torch.roll(bnd, -1, dims=0) - bnd
    normals = torch.stack([edges[:, 1], -edges[:, 0]], dim=-1)
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    # Determine winding: if CW, flip normals
    signed_area = (bnd[:, 0] * torch.roll(bnd[:, 1], -1) -
                   torch.roll(bnd[:, 0], -1) * bnd[:, 1]).sum()
    if signed_area < 0:
        normals = -normals

    bnd = bnd.detach()
    normals = normals.detach()

    def energy(x0: torch.Tensor) -> torch.Tensor:
        p_active, _ = _get_soft_types(x0, num_room_types)
        centers = x0[0, :, :2]  # (N, 2)

        # Signed distance from each center to each edge
        diff = centers.unsqueeze(1) - bnd.unsqueeze(0)  # (N, M, 2)
        signed_dist = (diff * normals.unsqueeze(0)).sum(dim=-1)  # (N, M)

        # Violation: positive when center is outside (signed_dist < 0)
        violation = F.relu(-signed_dist + margin)  # (N, M)

        # Max violation per slot (worst edge), weighted by p_active
        max_violation = violation.max(dim=-1).values  # (N,)
        return (max_violation * p_active).sum()

    return energy


def make_overlap_energy(
    num_room_types: int = 9,
    max_overlap_ratio: float = 0.3,
) -> Callable:
    """Penalize excessive overlap between bubbles.

    Weighted by p_active of both rooms in each pair.
    """
    def energy(x0: torch.Tensor) -> torch.Tensor:
        p_active, _ = _get_soft_types(x0, num_room_types)
        centers = x0[0, :, :2]
        radii = x0[0, :, 2].clamp(min=0.01)
        N = centers.shape[0]

        # Pairwise distances
        dist = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0))[0]
        r_sum = radii.unsqueeze(0) + radii.unsqueeze(1)
        r_min = torch.min(radii.unsqueeze(0), radii.unsqueeze(1))

        # Excess overlap
        overlap = r_sum - dist
        allowed = max_overlap_ratio * r_min
        excess = F.relu(overlap - allowed)

        # Weight by p_active of both slots
        pair_weight = p_active.unsqueeze(0) * p_active.unsqueeze(1)

        # Zero diagonal, upper triangle only
        mask = torch.triu(torch.ones(N, N, device=x0.device), diagonal=1)
        return (excess * pair_weight * mask).sum()

    return energy


def make_type_sharpness_energy(
    num_room_types: int = 9,
    temperature: float = 0.1,
) -> Callable:
    """Encourage type logits to be sharp (close to one-hot)."""
    def energy(x0: torch.Tensor) -> torch.Tensor:
        type_logits = x0[0, :, NUM_GEOM_DIMS:]
        probs = F.softmax(type_logits / temperature, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        return entropy.sum()

    return energy


def make_constraint_energy(
    constraints: torch.Tensor,
    constraint_mask: torch.Tensor,
    num_room_types: int = 9,
    num_constraint_types: int = 11,
) -> Callable:
    """Enforce architectural constraints from the constraint tensor."""
    valid = constraint_mask[0] > 0.5
    cons = constraints[0][valid]

    if len(cons) == 0:
        return lambda x0: torch.tensor(0.0, device=x0.device)

    # Decode constraint fields
    c_type_idx = cons[:, :num_constraint_types].argmax(dim=-1).detach()
    rt_a_idx = cons[:, num_constraint_types:num_constraint_types + num_room_types].argmax(dim=-1).detach()
    rt_b_idx = cons[:, num_constraint_types + num_room_types:num_constraint_types + 2 * num_room_types].argmax(dim=-1).detach()
    has_rt_a = (cons[:, num_constraint_types:num_constraint_types + num_room_types].sum(dim=-1) > 0.5).detach()
    has_rt_b = (cons[:, num_constraint_types + num_room_types:num_constraint_types + 2 * num_room_types].sum(dim=-1) > 0.5).detach()
    c_values = cons[:, -1].detach()

    # Constraint type constants
    MIN_AREA, MAX_AREA = 0, 1
    MIN_COUNT, MAX_COUNT = 2, 3
    MUST_EXIST = 6
    MIN_ROOM_RADIUS = 10

    def energy(x0: torch.Tensor) -> torch.Tensor:
        p_active, room_probs = _get_soft_types(x0, num_room_types)
        radii = x0[0, :, 2].clamp(min=0.001)

        total = torch.tensor(0.0, device=x0.device)

        for k in range(len(cons)):
            ct = c_type_idx[k].item()
            rta = rt_a_idx[k].item()
            val = c_values[k].item()

            if ct in (MIN_AREA, MAX_AREA) and has_rt_a[k]:
                w = room_probs[:, rta] * p_active
                areas = 3.14159 * radii ** 2
                if ct == MIN_AREA:
                    total = total + (F.relu(val - areas) * w).sum()
                else:
                    total = total + (F.relu(areas - val) * w).sum()

            elif ct in (MIN_COUNT, MAX_COUNT) and has_rt_a[k]:
                soft_count = (room_probs[:, rta] * p_active).sum()
                if ct == MIN_COUNT:
                    total = total + F.relu(val - soft_count)
                else:
                    total = total + F.relu(soft_count - val)

            elif ct == MUST_EXIST and has_rt_a[k]:
                soft_count = (room_probs[:, rta] * p_active).sum()
                total = total + F.relu(1.0 - soft_count)

            elif ct == MIN_ROOM_RADIUS and has_rt_a[k]:
                w = room_probs[:, rta] * p_active
                total = total + (F.relu(val - radii) * w).sum()

        return total

    return energy


def build_energy_fns(
    boundary: torch.Tensor,
    boundary_mask: torch.Tensor,
    constraints: torch.Tensor,
    constraint_mask: torch.Tensor,
    num_room_types: int = 9,
    boundary_weight: float = 2.0,
    overlap_weight: float = 1.0,
    constraint_weight: float = 1.0,
    type_sharpness_weight: float = 0.5,
) -> List[Callable]:
    """Build all energy functions for guided sampling."""
    fns = []

    bnd_fn = make_boundary_energy(boundary, boundary_mask, num_room_types)
    fns.append(lambda x0, f=bnd_fn, w=boundary_weight: w * f(x0))

    ovl_fn = make_overlap_energy(num_room_types)
    fns.append(lambda x0, f=ovl_fn, w=overlap_weight: w * f(x0))

    ts_fn = make_type_sharpness_energy(num_room_types)
    fns.append(lambda x0, f=ts_fn, w=type_sharpness_weight: w * f(x0))

    con_fn = make_constraint_energy(constraints, constraint_mask, num_room_types)
    fns.append(lambda x0, f=con_fn, w=constraint_weight: w * f(x0))

    return fns
