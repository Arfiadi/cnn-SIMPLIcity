"""Generate publication-grade 3D neural network architecture diagram for EfficientNet-B0 V4 (Expert).

Version 4: Perfection pass - zero overlaps in Panel C, pristine vertical hierarchy,
crystal-clear typography, and harmonious visual balance across all 3 panels.
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
    angle_deg: float = 36.0,
    alpha: float = 0.90,
    label_top: str | None = None,
    label_sub: str | None = None,
    label_dim: str | None = None,
    top_offset: float = 0.16,
) -> tuple[float, float, float, float]:
    """Draw a 3D isometric cuboid representing a convolutional feature map tensor."""
    rad = math.radians(angle_deg)
    dx = d * math.cos(rad)
    dy = d * math.sin(rad)

    top_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=min(1.0, alpha + 0.12))
    front_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=alpha)
    side_color = plt.matplotlib.colors.to_rgba(facecolor, alpha=max(0.0, alpha - 0.20))

    # 1. Front face
    front = patches.Rectangle(
        (x, y), w, h,
        facecolor=front_color, edgecolor=edgecolor, linewidth=1.2, zorder=3
    )
    ax.add_patch(front)

    # 2. Top face
    top_pts = np.array([
        [x, y + h],
        [x + dx, y + h + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h],
    ])
    top = patches.Polygon(
        top_pts, closed=True, facecolor=top_color, edgecolor=edgecolor, linewidth=1.2, zorder=3
    )
    ax.add_patch(top)

    # 3. Right face
    side_pts = np.array([
        [x + w, y],
        [x + w + dx, y + dy],
        [x + w + dx, y + h + dy],
        [x + w, y + h],
    ])
    side = patches.Polygon(
        side_pts, closed=True, facecolor=side_color, edgecolor=edgecolor, linewidth=1.2, zorder=3
    )
    ax.add_patch(side)

    # Top label (Stage / Name)
    if label_top:
        ax.text(
            x + w / 2 + dx / 2,
            y + h + dy + top_offset,
            label_top,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
            zorder=4,
        )

    # Sub-label (Operation)
    if label_sub:
        ax.text(
            x + w / 2,
            y - 0.20,
            label_sub,
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#1e293b",
            zorder=4,
        )

    # Dimension label (C, H, W)
    if label_dim:
        ax.text(
            x + w / 2,
            y - 0.48,
            label_dim,
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="bold",
            color="#334155",
            zorder=4,
        )

    return (x, x + w + dx, y, y + h + dy)


def main() -> None:
    fig = plt.figure(figsize=(22, 11.2), dpi=300, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.18, wspace=0.14)

    ax_top = fig.add_subplot(gs[0, :])
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    for ax in [ax_top, ax_bottom_left, ax_bottom_right]:
        ax.set_facecolor("#ffffff")
        ax.axis("off")

    # =========================================================================
    # 1. TOP PANEL: 3D ISOMETRIC FEATURE MAP PROGRESSION
    # =========================================================================
    ax_top.set_xlim(-0.2, 21.0)
    ax_top.set_ylim(-1.0, 4.4)

    header_box = patches.FancyBboxPatch(
        (0.0, 3.65), 20.8, 0.70,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.0
    )
    ax_top.add_patch(header_box)
    ax_top.text(
        0.3, 4.08,
        "A. EfficientNet-B0 Backbone Pipeline: 3D Isometric Tensor Progression",
        fontsize=13.0, fontweight="bold", color="#0f172a"
    )
    ax_top.text(
        0.3, 3.78,
        "Native Compound Scaling (240x240) | Letterbox ImageNet Mean Fill | 7 MBConv Stages | 5.3M Parameters | FLOPs: 0.39G",
        fontsize=9.2, fontweight="bold", color="#475569"
    )

    tensor_stages = [
        ("Input", "Letterbox Pad", "(3, 240, 240)", 0.24, 2.6, 1.25, "#94a3b8"),
        ("Stem", "Conv3x3 (s=2)", "(32, 120, 120)", 0.44, 2.0, 1.10, "#38bdf8"),
        ("Stage 1", "MBConv1 (x1)", "(16, 120, 120)", 0.38, 2.0, 1.10, "#0284c7"),
        ("Stage 2", "MBConv6 (x2)", "(24, 60, 60)", 0.48, 1.55, 0.85, "#0369a1"),
        ("Stage 3", "MBConv6 (x2)", "(40, 30, 30)", 0.58, 1.20, 0.70, "#0284c7"),
        ("Stage 4", "MBConv6 (x3)", "(80, 15, 15)", 0.75, 0.95, 0.55, "#0369a1"),
        ("Stage 5", "MBConv6 (x3)", "(112, 15, 15)", 0.88, 0.95, 0.55, "#0284c7"),
        ("Stage 6", "MBConv6 (x4)", "(192, 8, 8)", 1.10, 0.65, 0.40, "#0369a1"),
        ("Stage 7", "MBConv6 (x1)", "(320, 8, 8)", 1.30, 0.65, 0.40, "#075985"),
        ("Head Conv", "Conv 1x1", "(1280, 8, 8)", 1.75, 0.65, 0.40, "#4f46e5"),
        ("GAP", "AdaptiveAvgPool", "(1280, 1, 1)", 1.75, 0.24, 0.20, "#10b981"),
    ]

    x_curr = 0.5
    prev_pos = None

    for i, (name, subtext, dims, w, h, d, col) in enumerate(tensor_stages):
        y_pos = 1.25 - (h / 2.0)
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
            label_sub=subtext,
            label_dim=dims,
            top_offset=0.16,
        )

        if prev_pos is not None:
            px, py, pw, ph, pd = prev_pos
            ax_top.annotate(
                "",
                xy=(x_curr - 0.12, 1.25),
                xytext=(px + pw + pd * math.cos(math.radians(36)) + 0.08, 1.25),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#64748b",
                    lw=1.6,
                    mutation_scale=12,
                ),
                zorder=2,
            )

        prev_pos = (x_curr, y_pos, w, h, d)
        x_curr += w + d * math.cos(math.radians(36)) + 0.58

    gap_center_x = prev_pos[0] + prev_pos[2] / 2
    ax_top.annotate(
        "Feeds into Classification Head (Panel C)",
        xy=(gap_center_x, -0.65), xytext=(gap_center_x, 0.35),
        ha="center", va="center",
        fontsize=8.0, fontweight="bold", color="#059669",
        arrowprops=dict(
            arrowstyle="->",
            color="#10b981",
            lw=2.0,
            linestyle="--",
            mutation_scale=13,
        ),
    )

    # =========================================================================
    # 2. BOTTOM-LEFT PANEL: DEEP ZOOM INTO MBCONV BLOCK WITH SQUEEZE-AND-EXCITATION
    # =========================================================================
    ax_bottom_left.set_xlim(-0.2, 11.2)
    ax_bottom_left.set_ylim(-1.3, 4.6)

    box_b = patches.FancyBboxPatch(
        (0.0, 3.85), 11.0, 0.65,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#fff7ed", edgecolor="#fed7aa", linewidth=1.0
    )
    ax_bottom_left.add_patch(box_b)
    ax_bottom_left.text(
        0.3, 4.25,
        "B. MBConv Micro-Architecture: Inverted Residual + Squeeze-and-Excitation",
        fontsize=11.5, fontweight="bold", color="#9a3412"
    )
    ax_bottom_left.text(
        0.3, 3.98,
        "Mobile Inverted Bottleneck (Expansion E=6) with Depthwise Conv & SE Recalibration (r=0.25)",
        fontsize=8.5, fontweight="semibold", color="#c2410c"
    )

    container = patches.FancyBboxPatch(
        (0.0, -1.05), 11.0, 4.75,
        boxstyle="round,pad=0.2,rounding_size=0.25",
        facecolor="#fcfcfd", edgecolor="#e2e8f0", linewidth=1.5
    )
    ax_bottom_left.add_patch(container)

    sub_blocks = [
        ("Input Tensor\nC channels", 0.35, 0.85, 1.25, 1.55, "#cbd5e1", "#475569"),
        ("1x1 Conv + BN + SiLU\n(Expansion x6)\n6C channels", 2.05, 0.85, 1.55, 1.55, "#fed7aa", "#c2410c"),
        ("Depthwise 3x3/5x5\nConv + BN + SiLU\n(Spatial Feature)", 3.95, 0.85, 1.65, 1.55, "#fef08a", "#a16207"),
        ("Squeeze & Excitation\nGAP -> FC(C/4) ->\nSiLU -> FC(6C) -> Sigmoid", 5.95, 0.65, 1.95, 1.95, "#bbf7d0", "#15803d"),
        ("1x1 Conv + BN\n(Linear Bottleneck)\nC_out channels", 8.25, 0.85, 1.55, 1.55, "#fed7aa", "#c2410c"),
    ]

    for title, bx, by, bw, bh, bcol, tcol in sub_blocks:
        box = patches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bcol, edgecolor="#475569", linewidth=1.2, zorder=3
        )
        ax_bottom_left.add_patch(box)
        ax_bottom_left.text(
            bx + bw / 2, by + bh / 2,
            title,
            ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#0f172a", zorder=4
        )

    for i in range(len(sub_blocks) - 1):
        x1 = sub_blocks[i][1] + sub_blocks[i][3]
        x2 = sub_blocks[i + 1][1]
        ax_bottom_left.annotate(
            "", xy=(x2, 1.62), xytext=(x1, 1.62),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6, mutation_scale=10),
            zorder=2
        )

    add_x = 10.15
    add_y = 1.62
    add_circle = patches.Circle((add_x, add_y), 0.24, facecolor="#ea580c", edgecolor="#9a3412", linewidth=1.5, zorder=5)
    ax_bottom_left.add_patch(add_circle)
    ax_bottom_left.text(add_x, add_y, "+", ha="center", va="center", fontsize=14, fontweight="bold", color="#ffffff", zorder=6)

    ax_bottom_left.annotate(
        "", xy=(add_x - 0.24, add_y), xytext=(8.25 + 1.55, add_y),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6, mutation_scale=10),
        zorder=2
    )

    skip_path = patches.FancyArrowPatch(
        (0.95, 2.40), (add_x, 1.90),
        connectionstyle="arc3,rad=-0.36",
        color="#ea580c", lw=2.2, linestyle="--",
        arrowstyle="-|>", mutation_scale=13, zorder=4
    )
    ax_bottom_left.add_patch(skip_path)
    
    ax_bottom_left.text(
        5.4, 3.25,
        "Identity Residual Connection: x + F(x)\n(Active when Stride=1 & C_in == C_out)",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#ea580c",
        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.15", facecolor="#ffffff", edgecolor="#fed7aa", lw=1.0),
        zorder=6
    )

    ax_bottom_left.text(
        5.5, -0.65,
        "Key Advantage: Depthwise convolutions reduce FLOPs by 8-9x compared to standard conv,\nwhile Squeeze-and-Excitation dynamically recalibrates channel-wise feature dependencies.",
        ha="center", va="center", fontsize=7.6, color="#475569",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="#f8fafc", edgecolor="#e2e8f0", lw=0.8)
    )

    # =========================================================================
    # 3. BOTTOM-RIGHT PANEL: FULLY CONNECTED NEURAL NETWORK GRAPH (Enhanced Head)
    # =========================================================================
    ax_bottom_right.set_xlim(-0.2, 11.2)
    ax_bottom_right.set_ylim(-1.3, 4.6)

    box_c = patches.FancyBboxPatch(
        (0.0, 3.85), 11.0, 0.65,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#ecfdf5", edgecolor="#a7f3d0", linewidth=1.0
    )
    ax_bottom_right.add_patch(box_c)
    ax_bottom_right.text(
        0.3, 4.25,
        "C. Custom Enhanced Classification Head: Two-Stage Non-Linear MLP",
        fontsize=11.5, fontweight="bold", color="#065f46"
    )
    ax_bottom_right.text(
        0.3, 3.98,
        "BatchNorm1d(1280) -> Dropout(0.3) -> Linear(1280, 256) -> SiLU -> BatchNorm1d(256) -> Dropout(0.2) -> Linear(256, 10)",
        fontsize=8.0, fontweight="semibold", color="#047857"
    )

    container_head = patches.FancyBboxPatch(
        (0.0, -1.05), 11.0, 4.75,
        boxstyle="round,pad=0.2,rounding_size=0.25",
        facecolor="#fcfcfd", edgecolor="#e2e8f0", linewidth=1.5
    )
    ax_bottom_right.add_patch(container_head)

    x_in = 1.3
    x_hidden = 4.8
    x_out = 8.2

    n_in_nodes = 7
    n_hid_nodes = 6
    n_out_nodes = 10

    # Elevated vertical range to leave plenty of room at bottom
    y_in_nodes = np.linspace(3.0, 0.5, n_in_nodes)
    y_hid_nodes = np.linspace(2.8, 0.7, n_hid_nodes)
    y_out_nodes = np.linspace(3.2, 0.2, n_out_nodes)

    # Synaptic connections (Input -> Hidden)
    for y1 in y_in_nodes:
        for y2 in y_hid_nodes:
            ax_bottom_right.plot([x_in, x_hidden], [y1, y2], color="#cbd5e1", lw=0.55, alpha=0.55, zorder=1)

    # Synaptic connections (Hidden -> Output)
    for y1 in y_hid_nodes:
        for y2 in y_out_nodes:
            ax_bottom_right.plot([x_hidden, x_out], [y1, y2], color="#86efac", lw=0.60, alpha=0.65, zorder=1)

    # Draw Input Nodes (GAP Features)
    for idx, y in enumerate(y_in_nodes):
        c = patches.Circle((x_in, y), 0.13, facecolor="#38bdf8", edgecolor="#0284c7", linewidth=1.2, zorder=3)
        ax_bottom_right.add_patch(c)
    ax_bottom_right.text(x_in, 3.35, "GAP Features\n(1280 dims)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0284c7")
    ax_bottom_right.text(
        x_in, 0.05,
        "BN1d(1280)\n+ Drop(0.3)",
        ha="center", va="top", fontsize=7.4, fontweight="bold", color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.8),
        zorder=4
    )

    # Draw Hidden Nodes (Intermediate 256 Layer)
    for idx, y in enumerate(y_hid_nodes):
        c = patches.Circle((x_hidden, y), 0.14, facecolor="#a7f3d0", edgecolor="#059669", linewidth=1.2, zorder=3)
        ax_bottom_right.add_patch(c)
    ax_bottom_right.text(x_hidden, 3.35, "Hidden Projection\n(256 neurons + SiLU)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#059669")
    ax_bottom_right.text(
        x_hidden, 0.05,
        "BN1d(256)\n+ Drop(0.2)",
        ha="center", va="top", fontsize=7.4, fontweight="bold", color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.8),
        zorder=4
    )

    # Draw Output Nodes & Class Labels with pill tags
    class_names = [
        ("People", "#fef3c7", "#d97706"),
        ("Beach", "#dbeafe", "#2563eb"),
        ("Building", "#f3e8ff", "#9333ea"),
        ("Bus", "#ffedd5", "#ea580c"),
        ("Dinosaur", "#dcfce7", "#16a34a"),
        ("Elephant", "#e0e7ff", "#4f46e5"),
        ("Flower", "#fce7f3", "#db2777"),
        ("Horse", "#ccfbf1", "#0d9488"),
        ("Mountain", "#e2e8f0", "#475569"),
        ("Food", "#fee2e2", "#dc2626"),
    ]

    for idx, (y, (cname, bg_col, border_col)) in enumerate(zip(y_out_nodes, class_names)):
        c = patches.Circle((x_out, y), 0.11, facecolor="#fde047", edgecolor="#ca8a04", linewidth=1.2, zorder=3)
        ax_bottom_right.add_patch(c)
        ax_bottom_right.text(
            x_out + 0.28, y,
            f"  {idx}: {cname}  ",
            ha="left", va="center",
            fontsize=7.4, fontweight="bold", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.2", facecolor=bg_col, edgecolor=border_col, lw=0.9),
            zorder=4
        )

    ax_bottom_right.text(x_out + 0.9, 3.52, "Class Logits (10 Categories)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#ca8a04")

    # Bottom summary tag placed with comfortable clearance
    ax_bottom_right.text(
        5.5, -0.68,
        "Regularization: Two-stage Dropout (0.3 -> 0.2) and dual BatchNorm prevent overfitting on N=500,\nwhile intermediate 256 projection captures non-linear semantic boundaries.",
        ha="center", va="center", fontsize=7.6, color="#475569",
        bbox=dict(boxstyle="square,pad=0.25", facecolor="#f8fafc", edgecolor="#e2e8f0", lw=0.8),
        zorder=4
    )

    output_path = Path("reports/figures/efficientnet_b0_architecture.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Version 4 architecture diagram saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
