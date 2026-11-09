"""Generate publication-grade 3D neural network architecture diagram.

Visualizes the audited EfficientNet-B0 V4 (Expert) model pipeline used in the SIMPLIcity project:
1. Aspect-Ratio Preserving Letterbox Preprocessing (240x240)
2. 3D Isometric Tensor Feature Maps across all 7 MBConv Stages + Stem
3. Deep Zoom: Internal MBConv Micro-Architecture (Expansion -> DW-Conv -> Squeeze-and-Excitation -> Projection -> Skip)
4. Fully Connected Neural Network Graph for the Custom Two-Stage Classification Head (1280 -> 256 -> 10).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_isometric_cuboid(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    d: float,
    facecolor: str = "#2b5c8f",
    edgecolor: str = "#1a365d",
    angle_deg: float = 38.0,
    alpha: float = 0.88,
    label_top: str | None = None,
    label_bottom: str | None = None,
    label_dim: str | None = None,
) -> None:
    """Draw a 3D isometric cuboid representing a convolutional feature map tensor."""
    rad = math.radians(angle_deg)
    dx = d * math.cos(rad)
    dy = d * math.sin(rad)

    top_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=min(1.0, alpha + 0.10))
    front_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=alpha)
    side_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=max(0.0, alpha - 0.18))

    # 1. Front face
    front = patches.Rectangle((x, y), w, h, facecolor=front_color, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(front)

    # 2. Top face
    top_pts = np.array([
        [x, y + h],
        [x + dx, y + h + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h],
    ])
    top = patches.Polygon(top_pts, closed=True, facecolor=top_color, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(top)

    # 3. Right face
    side_pts = np.array([
        [x + w, y],
        [x + w + dx, y + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h],
    ])
    side = patches.Polygon(side_pts, closed=True, facecolor=side_color, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(side)

    # Labels
    if label_top:
        ax.text(
            x + w / 2 + dx / 2,
            y + h + dy + 0.16,
            label_top,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
        )
    if label_bottom:
        ax.text(
            x + w / 2,
            y - 0.18,
            label_bottom,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#334155",
        )
    if label_dim:
        ax.text(
            x + w / 2,
            y - 0.44,
            label_dim,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#64748b",
        )


def main() -> None:
    fig = plt.figure(figsize=(22, 12), dpi=300, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.16)

    ax_top = fig.add_subplot(gs[0, :])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    for ax in [ax_top, ax_bottom_left, ax_bottom_right]:
        ax.set_facecolor("#ffffff")
        ax.axis("off")

    # =========================================================================
    # 1. TOP PANEL: 3D ISOMETRIC FEATURE MAP PROGRESSION
    # =========================================================================
    ax_top.set_xlim(-0.5, 20.0)
    ax_top.set_ylim(-1.6, 5.0)

    ax_top.text(
        0.0, 4.5,
        "A. EfficientNet-B0 Backbone: 3D Isometric Tensor Feature Map Progression",
        fontsize=14, fontweight="bold", color="#0f172a"
    )
    ax_top.text(
        0.0, 4.15,
        "Native Compound Scaling (240x240) with Aspect-Ratio Letterbox Padding & 7-Stage MBConv Pipeline",
        fontsize=10.5, color="#475569"
    )

    tensor_stages = [
        ("Input", "Letterbox Pad", "(3, 240, 240)", 0.22, 2.7, 1.3, "#94a3b8"),
        ("Stem", "Conv3x3 (s=2)", "(32, 120, 120)", 0.42, 2.1, 1.1, "#38bdf8"),
        ("Stage 1", "MBConv1 (x1)", "(16, 120, 120)", 0.38, 2.1, 1.1, "#0284c7"),
        ("Stage 2", "MBConv6 (x2)", "(24, 60, 60)", 0.48, 1.6, 0.85, "#0369a1"),
        ("Stage 3", "MBConv6 (x2)", "(40, 30, 30)", 0.58, 1.25, 0.70, "#0284c7"),
        ("Stage 4", "MBConv6 (x3)", "(80, 15, 15)", 0.75, 0.95, 0.55, "#0369a1"),
        ("Stage 5", "MBConv6 (x3)", "(112, 15, 15)", 0.85, 0.95, 0.55, "#0284c7"),
        ("Stage 6", "MBConv6 (x4)", "(192, 8, 8)", 1.05, 0.65, 0.40, "#0369a1"),
        ("Stage 7", "MBConv6 (x1)", "(320, 8, 8)", 1.25, 0.65, 0.40, "#075985"),
        ("Head Conv", "Conv 1x1", "(1280, 8, 8)", 1.70, 0.65, 0.40, "#4f46e5"),
        ("GAP", "AdaptiveAvgPool", "(1280, 1, 1)", 1.70, 0.22, 0.20, "#10b981"),
    ]

    x_curr = 0.4
    prev_pos = None

    for i, (name, subtext, dims, w, h, d, col) in enumerate(tensor_stages):
        y_pos = 1.0 - (h / 2.0)
        draw_isometric_cuboid(
            ax_top,
            x=x_curr,
            y=y_pos,
            w=w,
            h=h,
            d=d,
            facecolor=col,
            edgecolor="#1e293b",
            label_top=name,
            label_bottom=subtext,
            label_dim=dims,
        )

        if prev_pos is not None:
            px, py, pw, ph, pd = prev_pos
            ax_top.annotate(
                "",
                xy=(x_curr - 0.12, 1.0),
                xytext=(px + pw + pd * math.cos(math.radians(38)) + 0.08, 1.0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#64748b",
                    lw=1.5,
                    mutation_scale=11,
                ),
            )

        prev_pos = (x_curr, y_pos, w, h, d)
        x_curr += w + d * math.cos(math.radians(38)) + 0.55

    # =========================================================================
    # 2. BOTTOM-LEFT PANEL: MBCONV BLOCK WITH SQUEEZE-AND-EXCITATION
    # =========================================================================
    ax_bottom_left.set_xlim(-0.3, 11.2)
    ax_bottom_left.set_ylim(-1.5, 5.0)

    ax_bottom_left.text(
        0.0, 4.65,
        "B. MBConv Block Architecture (Inverted Residual + Squeeze-and-Excitation)",
        fontsize=12.5, fontweight="bold", color="#0f172a"
    )
    ax_bottom_left.text(
        0.0, 4.30,
        "Depthwise Separable Convolution with Expansion E=6 and SE Recalibration r=0.25",
        fontsize=9.5, color="#475569"
    )

    container = patches.FancyBboxPatch(
        (0.2, -0.9), 10.7, 4.9,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5
    )
    ax_bottom_left.add_patch(container)

    sub_blocks = [
        ("Input Tensor\nC channels", 0.5, 0.9, 1.25, 1.5, "#cbd5e1"),
        ("1x1 Conv + BN\n(Expansion x6)\n6C channels", 2.15, 0.9, 1.45, 1.5, "#fed7aa"),
        ("Depthwise 3x3/5x5\nConv + BN + SiLU\n(Spatial Filter)", 3.95, 0.9, 1.55, 1.5, "#fde68a"),
        ("Squeeze & Excitation\nGAP -> FC(C/4) ->\nSiLU -> FC(6C) -> Sigmoid", 5.85, 0.75, 1.85, 1.8, "#bbf7d0"),
        ("1x1 Conv + BN\n(Linear Projection)\nC_out channels", 8.05, 0.9, 1.45, 1.5, "#fed7aa"),
    ]

    for title, bx, by, bw, bh, bcol in sub_blocks:
        box = patches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bcol, edgecolor="#475569", linewidth=1.2
        )
        ax_bottom_left.add_patch(box)
        ax_bottom_left.text(
            bx + bw / 2, by + bh / 2,
            title,
            ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#0f172a"
        )

    for i in range(len(sub_blocks) - 1):
        x1 = sub_blocks[i][1] + sub_blocks[i][3]
        x2 = sub_blocks[i + 1][1]
        ax_bottom_left.annotate(
            "", xy=(x2, 1.65), xytext=(x1, 1.65),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, mutation_scale=10)
        )

    # Addition Circle (positioned after projection)
    add_x = 9.9
    add_y = 1.65
    add_circle = patches.Circle((add_x, add_y), 0.22, facecolor="#ea580c", edgecolor="#9a3412", linewidth=1.5)
    ax_bottom_left.add_patch(add_circle)
    ax_bottom_left.text(add_x, add_y, "+", ha="center", va="center", fontsize=13, fontweight="bold", color="#ffffff")

    # Arrow from 1x1 projection to addition
    ax_bottom_left.annotate(
        "", xy=(add_x - 0.22, add_y), xytext=(8.05 + 1.45, add_y),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, mutation_scale=10)
    )

    # Arrow from addition to output
    ax_bottom_left.annotate(
        "", xy=(10.6, add_y), xytext=(add_x + 0.22, add_y),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, mutation_scale=10)
    )
    ax_bottom_left.text(10.65, add_y, "Output", ha="left", va="center", fontsize=8.0, fontweight="bold", color="#334155")

    # Residual Skip-Connection Loop
    skip_path = patches.FancyArrowPatch(
        (1.1, 2.45), (add_x, 2.05),
        connectionstyle="arc3,rad=-0.38",
        color="#ea580c", lw=2.0, linestyle="--",
        arrowstyle="-|>", mutation_scale=13
    )
    ax_bottom_left.add_patch(skip_path)
    ax_bottom_left.text(
        5.5, 3.65,
        "Identity Residual Connection: x + F(x) (Active when Stride=1 & C_in == C_out)",
        ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#ea580c"
    )

    # =========================================================================
    # 3. BOTTOM-RIGHT PANEL: FULLY CONNECTED NEURAL NETWORK GRAPH (Enhanced Head)
    # =========================================================================
    ax_bottom_right.set_xlim(-0.5, 10.5)
    ax_bottom_right.set_ylim(-1.5, 5.0)

    ax_bottom_right.text(
        0.0, 4.65,
        "C. Custom Enhanced Classification Head (Two-Stage Non-Linear MLP)",
        fontsize=12.5, fontweight="bold", color="#0f172a"
    )
    ax_bottom_right.text(
        0.0, 4.30,
        "BatchNorm1d(1280) -> Dropout(0.3) -> Linear(1280, 256) -> SiLU -> BatchNorm1d(256) -> Dropout(0.2) -> Linear(256, 10)",
        fontsize=8.2, color="#475569"
    )

    container_head = patches.FancyBboxPatch(
        (0.2, -0.9), 9.8, 4.9,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.5
    )
    ax_bottom_right.add_patch(container_head)

    x_in = 1.4
    x_hidden = 4.8
    x_out = 8.1

    n_in_nodes = 7
    n_hid_nodes = 6
    n_out_nodes = 10

    y_in_nodes = np.linspace(2.8, -0.2, n_in_nodes)
    y_hid_nodes = np.linspace(2.6, 0.0, n_hid_nodes)
    y_out_nodes = np.linspace(3.1, -0.5, n_out_nodes)

    # Synaptic connections (Input -> Hidden)
    for y1 in y_in_nodes:
        for y2 in y_hid_nodes:
            ax_bottom_right.plot([x_in, x_hidden], [y1, y2], color="#cbd5e1", lw=0.45, alpha=0.50)

    # Synaptic connections (Hidden -> Output)
    for y1 in y_hid_nodes:
        for y2 in y_out_nodes:
            ax_bottom_right.plot([x_hidden, x_out], [y1, y2], color="#86efac", lw=0.50, alpha=0.60)

    # Draw Input Nodes
    for idx, y in enumerate(y_in_nodes):
        c = patches.Circle((x_in, y), 0.12, facecolor="#38bdf8", edgecolor="#0284c7", linewidth=1.2)
        ax_bottom_right.add_patch(c)
    ax_bottom_right.text(x_in, 3.20, "GAP Features\n(1280 dims)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0284c7")
    ax_bottom_right.text(x_in, -0.58, "BN1d + Drop(0.3)", ha="center", va="top", fontsize=7.2, fontweight="bold", color="#64748b")

    # Draw Hidden Nodes
    for idx, y in enumerate(y_hid_nodes):
        c = patches.Circle((x_hidden, y), 0.13, facecolor="#a7f3d0", edgecolor="#059669", linewidth=1.2)
        ax_bottom_right.add_patch(c)
    ax_bottom_right.text(x_hidden, 3.20, "Hidden Layer\n(256 neurons + SiLU)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#059669")
    ax_bottom_right.text(x_hidden, -0.58, "BN1d + Drop(0.2)", ha="center", va="top", fontsize=7.2, fontweight="bold", color="#64748b")

    # Draw Output Nodes & Class Labels
    class_names = [
        "People", "Beach", "Building", "Bus", "Dinosaur",
        "Elephant", "Flower", "Horse", "Mountain", "Food"
    ]
    for idx, (y, cname) in enumerate(zip(y_out_nodes, class_names)):
        c = patches.Circle((x_out, y), 0.11, facecolor="#fde047", edgecolor="#ca8a04", linewidth=1.2)
        ax_bottom_right.add_patch(c)
        ax_bottom_right.text(
            x_out + 0.22, y,
            f"c_{idx}: {cname}",
            ha="left", va="center",
            fontsize=7.0, fontweight="bold", color="#1e293b"
        )
    ax_bottom_right.text(x_out + 0.5, 3.45, "Class Logits (10)", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#ca8a04")

    # Save High-Resolution Artifact
    output_path = Path("reports/figures/efficientnet_b0_architecture.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Refined architecture diagram saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
