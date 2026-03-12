"""Visualization of bubble diagrams."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Room type index -> name mapping
ROOM_TYPE_NAMES: Dict[int, str] = {
    0: "Bedroom",
    1: "Living",
    2: "Kitchen",
    3: "Dining",
    4: "Corridor",
    5: "Stairs",
    6: "Storeroom",
    7: "Bathroom",
    8: "Balcony",
}

# Room type name -> color mapping
ROOM_COLORS: Dict[str, str] = {
    "Bedroom": "blue",
    "Living": "orange",
    "Kitchen": "red",
    "Dining": "teal",
    "Corridor": "gray",
    "Stairs": "yellow",
    "Storeroom": "tan",
    "Bathroom": "green",
    "Balcony": "purple",
}


def _get_color(room_type: int) -> str:
    """Return the color for a given room type index."""
    name = ROOM_TYPE_NAMES.get(room_type, "Unknown")
    return ROOM_COLORS.get(name, "black")


def _get_name(room_type: int) -> str:
    """Return the name for a given room type index."""
    return ROOM_TYPE_NAMES.get(room_type, f"Type{room_type}")


def plot_bubbles(
    bubbles: List[Tuple[float, float, float, int]],
    boundary: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """Plot a bubble diagram with colored circles and room labels.

    Parameters
    ----------
    bubbles : list of (cx, cy, radius, room_type)
    boundary : ndarray of shape (N, 2), optional
        Boundary polygon vertices. If provided, drawn as an outline.
    title : str
        Plot title.
    save_path : str, optional
        If given, save the figure to this path and close it.
        Otherwise call plt.show().
    figsize : tuple of (width, height)
        Figure size in inches.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw boundary polygon outline
    if boundary is not None and len(boundary) >= 3:
        boundary_closed = np.vstack([boundary, boundary[0]])
        ax.plot(
            boundary_closed[:, 0],
            boundary_closed[:, 1],
            color="black",
            linewidth=2,
            linestyle="-",
            label="_boundary",
        )

    # Track which types are present for legend
    present_types: Dict[int, str] = {}

    # Draw circles
    for cx, cy, r, rt in bubbles:
        color = _get_color(rt)
        name = _get_name(rt)
        present_types[rt] = name

        circle = plt.Circle(
            (cx, cy),
            r,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.4,
        )
        ax.add_patch(circle)

        # Label at center
        ax.text(
            cx,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
        )

    # Legend for present types
    legend_handles = []
    for rt in sorted(present_types.keys()):
        name = present_types[rt]
        color = ROOM_COLORS.get(name, "black")
        patch = mpatches.Patch(color=color, alpha=0.4, label=name)
        legend_handles.append(patch)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
