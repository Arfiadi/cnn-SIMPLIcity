"""Generate publication-grade visual error analysis panel for EfficientNet-B0 V4 (Expert).

Inspects the exactly 4 misclassified samples (out of 200 test images) with side-by-side
thumbnails, ground truth vs predicted labels, confidence scores, and root-cause annotations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def main() -> None:
    samples = [
        {
            "filename": "295.jpg",
            "true_label": "Building",
            "pred_label": "People",
            "conf": 0.7616,
            "category": "Label Ambiguity (Pompeii Human Cast)",
            "description": "Anatomical plaster cast of a human victim\nof Pompeii. Ground truth labeled as 'Building'\ndue to excavation site location.",
            "color": "#dc2626",
        },
        {
            "filename": "290.jpg",
            "true_label": "Building",
            "pred_label": "People",
            "conf": 0.4607,
            "category": "Label Ambiguity (Classical Statue)",
            "description": "Outdoor Roman marble statue of a draped human\nfigure with toga. Model recognizes human posture\nand fabric folds (low confidence 0.46).",
            "color": "#ea580c",
        },
        {
            "filename": "291.jpg",
            "true_label": "Building",
            "pred_label": "People",
            "conf": 0.9274,
            "category": "Spectral / Texture Bias",
            "description": "Herculaneum room ruins with intense red\nterracotta and ochre plaster. Activates skin/fabric\ncolor features of African tribal People category.",
            "color": "#d97706",
        },
        {
            "filename": "883.jpg",
            "true_label": "Mountain",
            "pred_label": "Beach",
            "conf": 0.5530,
            "category": "Visual Landscape Ambiguity",
            "description": "Glacial lake with floating white icebergs in the\nforeground water, mimicking ocean surf / foam\nalong a beach shoreline (near boundary: 0.55).",
            "color": "#0284c7",
        },
    ]

    target_aspect_size = (384, 256)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.2), dpi=300, facecolor="#ffffff")
    fig.suptitle(
        "EfficientNet-B0 V4 (Expert): Microscopic Analysis of Remaining 4 Test Errors (4 / 200)",
        fontsize=15, fontweight="bold", y=0.98, color="#0f172a"
    )

    for idx, (ax, s) in enumerate(zip(axes, samples)):
        ax.set_facecolor("#ffffff")
        img_path = Path("Dataset/SIMPLIcity/test") / s["filename"]
        if not img_path.exists():
            img_path = Path("C:/Users/USER/.gemini/antigravity/brain/34a2f330-2af5-4865-ba97-8b4bf3ff57e4/misclassified_v4") / s["filename"]

        raw_img = Image.open(img_path)
        # Pad to uniform aspect ratio canvas so all image boxes and text line up identically
        fitted_img = ImageOps.pad(raw_img, target_aspect_size, color=(240, 240, 240))

        ax.imshow(fitted_img)
        ax.axis("off")

        # Top tag
        title_text = f"Sample [{idx + 1}/4]: {s['filename']}"
        ax.set_title(title_text, fontsize=11, fontweight="bold", pad=8, color="#1e293b")

        # Annotations text below image
        info_text = (
            f"True: {s['true_label']}  ->  Pred: {s['pred_label']}\n"
            f"Confidence: {s['conf']:.4f}\n"
            f"Type: {s['category']}\n\n"
            f"{s['description']}"
        )

        ax.text(
            0.5, -0.06, info_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.5,
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.5,rounding_size=0.2",
                facecolor="#f8fafc",
                edgecolor=s["color"],
                linewidth=1.5,
            )
        )

    plt.subplots_adjust(top=0.88, bottom=0.32, wspace=0.20)
    out_path = Path("reports/figures/misclassified_samples_panel.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Error analysis panel successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
