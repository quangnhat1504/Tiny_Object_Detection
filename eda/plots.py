"""Cac ham ve bieu do EDA cho dataset TinyPerson. Luu ra file PNG."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SPLIT_COLORS = {"train": "#4C72B0", "valid": "#DD8452", "test": "#55A868"}


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def plot_split_distribution(images, instances, summary, fig_dir, splits):
    """So anh va so object theo tung split."""
    n_img = [summary["splits"][s]["n_images"] for s in splits]
    n_inst = [summary["splits"][s]["n_instances"] for s in splits]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = [SPLIT_COLORS[s] for s in splits]
    for ax, vals, title in [(axes[0], n_img, "So luong anh / split"),
                            (axes[1], n_inst, "So luong object / split")]:
        bars = ax.bar(splits, vals, color=colors)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%d")
        ax.set_ylabel("count")
    _save(fig, fig_dir / "01_split_distribution.png")


def plot_class_distribution(summary, fig_dir, classes, splits):
    """Phan bo class tong the va theo split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    tot = summary["total"]["class_counts"]
    bars = axes[0].bar(list(tot.keys()), list(tot.values()),
                       color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Phan bo class (toan dataset)")
    axes[0].bar_label(bars, fmt="%d")
    axes[0].set_ylabel("so object")

    x = np.arange(len(splits))
    width = 0.38
    for i, c in enumerate(classes):
        vals = [summary["splits"][s]["class_counts"][c] for s in splits]
        b = axes[1].bar(x + (i - 0.5) * width, vals, width, label=c)
        axes[1].bar_label(b, fmt="%d", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Phan bo class theo split")
    axes[1].legend()
    axes[1].set_ylabel("so object")
    _save(fig, fig_dir / "02_class_distribution.png")


def plot_size_distribution(instances, summary, fig_dir, size_bins, size_labels):
    """Histogram sqrt(area) + phan loai theo thang kich thuoc tiny/small."""
    sq = np.array([i["sqrt_area"] for i in instances])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    clip = np.clip(sq, 0, 120)
    axes[0].hist(clip, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].axvline(20, color="red", ls="--", lw=1, label="20 px (tiny)")
    axes[0].axvline(32, color="orange", ls="--", lw=1, label="32 px (small)")
    axes[0].set_xlabel("sqrt(area) [px]  (clip o 120)")
    axes[0].set_ylabel("so object")
    axes[0].set_title("Phan bo kich thuoc tuyet doi")
    axes[0].legend()

    counts = [summary["size_category_counts"][lbl] for lbl in size_labels]
    bars = axes[1].bar(size_labels, counts, color="#55A868")
    axes[1].bar_label(bars, fmt="%d", fontsize=8)
    axes[1].set_xlabel("sqrt(area) bin [px]")
    axes[1].set_ylabel("so object")
    axes[1].set_title("Phan loai theo thang kich thuoc")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, fig_dir / "03_size_distribution.png")


def plot_size_log(instances, fig_dir):
    """Histogram dien tich (px^2) thang log de thay duoi phan bo."""
    area = np.array([i["area_px"] for i in instances])
    area = area[area > 0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.logspace(np.log10(max(area.min(), 1)), np.log10(area.max()), 50)
    ax.hist(area, bins=bins, color="#8172B3", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Dien tich bbox [px^2] (log)")
    ax.set_ylabel("so object")
    ax.set_title("Phan bo dien tich bbox (thang log)")
    _save(fig, fig_dir / "04_area_log.png")


def plot_wh_scatter(instances, fig_dir, classes):
    """Scatter width vs height (px) tach theo class."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cmap = {classes[0]: "#4C72B0", classes[1] if len(classes) > 1 else "x": "#C44E52"}
    for c in classes:
        w = [i["bw_px"] for i in instances if i["class"] == c]
        h = [i["bh_px"] for i in instances if i["class"] == c]
        ax.scatter(w, h, s=6, alpha=0.35, label=c, color=cmap.get(c))
    lim = 100
    ax.plot([0, lim], [0, lim], "k--", lw=0.7, alpha=0.5, label="w=h")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("width [px]")
    ax.set_ylabel("height [px]")
    ax.set_title("Width vs Height (clip 100px)")
    ax.legend()
    _save(fig, fig_dir / "05_wh_scatter.png")


def plot_aspect_ratio(instances, fig_dir):
    """Phan bo ti le khung (w/h)."""
    ar = np.array([i["aspect"] for i in instances if i["aspect"] > 0])
    ar = np.clip(ar, 0, 4)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(ar, bins=50, color="#DD8452", edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="black", ls="--", lw=1, label="w=h")
    med = float(np.median(ar)) if ar.size else 0
    ax.axvline(med, color="red", ls="-", lw=1, label=f"median={med:.2f}")
    ax.set_xlabel("aspect ratio (w/h, clip 4)")
    ax.set_ylabel("so object")
    ax.set_title("Phan bo ti le khung bbox")
    ax.legend()
    _save(fig, fig_dir / "06_aspect_ratio.png")


def plot_objects_per_image(images, fig_dir, splits):
    """Phan bo so object tren moi anh."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    alln = [im["n_obj"] for im in images]
    hi = int(np.percentile(alln, 99)) if alln else 1
    hi = max(hi, 1)
    bins = np.arange(0, hi + 2) - 0.5
    for s in splits:
        vals = [im["n_obj"] for im in images if im["split"] == s]
        ax.hist(vals, bins=bins, alpha=0.6, label=s, color=SPLIT_COLORS[s])
    ax.set_xlabel(f"so object / anh (clip ~p99={hi})")
    ax.set_ylabel("so anh")
    ax.set_title("Phan bo so object tren moi anh")
    ax.set_xlim(-0.5, hi + 0.5)
    ax.legend()
    _save(fig, fig_dir / "07_objects_per_image.png")


def plot_center_heatmap(instances, fig_dir):
    """Heatmap vi tri tam object (normalized)."""
    cx = np.array([i["cx"] for i in instances])
    cy = np.array([i["cy"] for i in instances])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    h = ax.hist2d(cx, cy, bins=50, range=[[0, 1], [0, 1]], cmap="inferno")
    ax.invert_yaxis()  # goc toa do anh: y tang xuong duoi
    ax.set_xlabel("cx (normalized)")
    ax.set_ylabel("cy (normalized)")
    ax.set_title("Heatmap vi tri tam object")
    fig.colorbar(h[3], ax=ax, label="so object")
    _save(fig, fig_dir / "08_center_heatmap.png")


def plot_size_by_class(instances, fig_dir, classes):
    """Box/violin sqrt(area) theo class."""
    data = [[i["sqrt_area"] for i in instances if i["class"] == c] for c in classes]
    data = [d if d else [0] for d in data]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bp = ax.boxplot(data, labels=classes, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.axhline(20, color="red", ls="--", lw=1, label="20px (tiny)")
    ax.axhline(32, color="orange", ls="--", lw=1, label="32px (small)")
    ax.set_ylabel("sqrt(area) [px]")
    ax.set_title("Kich thuoc object theo class")
    ax.legend()
    _save(fig, fig_dir / "09_size_by_class.png")


def make_all(instances, images, summary, fig_dir, classes, splits,
             size_bins, size_labels):
    plot_split_distribution(images, instances, summary, fig_dir, splits)
    plot_class_distribution(summary, fig_dir, classes, splits)
    if instances:
        plot_size_distribution(instances, summary, fig_dir, size_bins, size_labels)
        plot_size_log(instances, fig_dir)
        plot_wh_scatter(instances, fig_dir, classes)
        plot_aspect_ratio(instances, fig_dir)
        plot_center_heatmap(instances, fig_dir)
        plot_size_by_class(instances, fig_dir, classes)
    plot_objects_per_image(images, fig_dir, splits)
