"""Sinh bao cao Markdown tu so lieu thong ke thuc te (summary dict)."""
from __future__ import annotations

from pathlib import Path


def _fmt(x, nd=1):
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def write_report(summary, out_path: Path, fig_dir: Path, classes, splits):
    rel = fig_dir.name  # "figures"
    t = summary["total"]
    L = []
    A = L.append

    A("# Bao cao EDA - Dataset TinyPerson\n")
    A("> Sinh tu dong boi `eda/eda_tinyperson.py`. Moi so lieu duoc tinh truc tiep "
      "tu file anh va nhan trong `data/`.\n")

    # 1. Tong quan
    A("## 1. Tong quan\n")
    A(f"- **Tong so anh**: {_fmt(t['n_images'])}")
    A(f"- **Tong so object (instance)**: {_fmt(t['n_instances'])}")
    A(f"- **So class**: {len(classes)} -> {', '.join(f'`{c}`' for c in classes)}")
    A(f"- **Anh khong co object (background)**: {_fmt(t['n_background'])}")
    avg = t['n_instances'] / t['n_images'] if t['n_images'] else 0
    A(f"- **Trung binh object / anh**: {avg:.2f}")
    A(f"- **Dinh dang nhan**: YOLO polygon (da chuyen sang bounding box de thong ke)\n")

    # 2. Phan chia split
    A("## 2. Phan chia train / valid / test\n")
    A("| Split | So anh | So object | Background | obj/anh (TB) | obj/anh (max) |")
    A("|-------|-------:|----------:|-----------:|-------------:|--------------:|")
    for s in splits:
        d = summary["splits"][s]
        A(f"| {s} | {_fmt(d['n_images'])} | {_fmt(d['n_instances'])} | "
          f"{_fmt(d['n_background'])} | {d['obj_per_img_mean']:.2f} | {d['obj_per_img_max']} |")
    A(f"| **Tong** | **{_fmt(t['n_images'])}** | **{_fmt(t['n_instances'])}** | "
      f"**{_fmt(t['n_background'])}** | - | - |\n")
    A(f"![Phan bo split]({rel}/01_split_distribution.png)\n")

    # 3. Phan bo class
    A("## 3. Phan bo class (mat can bang)\n")
    A("| Class | " + " | ".join(splits) + " | Tong | Ti le |")
    A("|-------|" + "------:|" * (len(splits) + 2))
    tot_all = t["n_instances"] or 1
    for c in classes:
        per = [summary["splits"][s]["class_counts"][c] for s in splits]
        ctot = t["class_counts"][c]
        A(f"| `{c}` | " + " | ".join(_fmt(v) for v in per) +
          f" | {_fmt(ctot)} | {100*ctot/tot_all:.1f}% |")
    A("")
    cc = t["class_counts"]
    if len(classes) == 2 and min(cc.values()) > 0:
        ratio = max(cc.values()) / min(cc.values())
        domc = max(cc, key=cc.get)
        A(f"> **Nhan xet**: lop `{domc}` chiem uu the, ti le mat can bang khoang "
          f"**{ratio:.1f}:1**. Can luu y khi train (vd: class weights / focal loss).\n")
    A(f"![Phan bo class]({rel}/02_class_distribution.png)\n")

    # 4. Kich thuoc - trong tam tiny object
    A("## 4. Kich thuoc object (dac thu tiny object)\n")
    A("Kich thuoc tinh bang `sqrt(dien tich bbox)` theo pixel tuyet doi.\n")
    A("| Chi so | Toan dataset |")
    A("|--------|-------------:|")
    A(f"| sqrt(area) trung binh | {t['sqrt_area_mean']:.1f} px |")
    A(f"| sqrt(area) trung vi | {t['sqrt_area_median']:.1f} px |")
    A(f"| sqrt(area) nho nhat | {t['sqrt_area_min']:.1f} px |")
    A(f"| sqrt(area) lon nhat | {t['sqrt_area_max']:.1f} px |")
    A(f"| Object \"tiny\" (< 20px) | **{t['pct_tiny_lt20']:.1f}%** |")
    A(f"| Object \"small\" (< 32px) | **{t['pct_small_lt32']:.1f}%** |\n")
    A("Phan loai theo thang kich thuoc (sqrt area):\n")
    A("| Bin (px) | So object |")
    A("|----------|----------:|")
    for lbl, cnt in summary["size_category_counts"].items():
        A(f"| {lbl} | {_fmt(cnt)} |")
    A("")
    A(f"> **Nhan xet**: {t['pct_small_lt32']:.0f}% object nho hon 32px - day dung la "
      f"bai toan tiny/small object detection. Cac model cau hinh mac dinh (anchor/stride lon) "
      f"se bo sot phan lon muc tieu; can chu y feature map do phan giai cao, anchor nho.\n")
    A(f"![Phan bo kich thuoc]({rel}/03_size_distribution.png)\n")
    A(f"![Dien tich log]({rel}/04_area_log.png)\n")
    A(f"![Kich thuoc theo class]({rel}/09_size_by_class.png)\n")

    # 5. Hinh dang
    A("## 5. Hinh dang & ti le khung\n")
    A(f"![Width vs Height]({rel}/05_wh_scatter.png)\n")
    A(f"![Aspect ratio]({rel}/06_aspect_ratio.png)\n")

    # 6. Phan bo khong gian & mat do
    A("## 6. Phan bo khong gian & mat do\n")
    A(f"![Object tren moi anh]({rel}/07_objects_per_image.png)\n")
    A(f"![Heatmap vi tri]({rel}/08_center_heatmap.png)\n")

    # 7. Kich thuoc anh
    A("## 7. Kich thuoc anh\n")
    A("| Split | Cac kich thuoc (WxH : so anh) |")
    A("|-------|------------------------------|")
    for s in splits:
        sizes = summary["splits"][s]["img_sizes"]
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
        txt = ", ".join(f"{k}:{v}" for k, v in top) or "-"
        A(f"| {s} | {txt} |")
    A("")

    # 8. Tom tat
    A("## 8. Tom tat & khuyen nghi\n")
    A(f"1. Dataset gom **{_fmt(t['n_images'])} anh / {_fmt(t['n_instances'])} object**, "
      f"2 lop nguoi (`dry-person`, `wet-swimmer`).")
    A(f"2. **Mat can bang lop ro ret** - can class weighting hoac oversampling lop thieu.")
    A(f"3. **~{t['pct_small_lt32']:.0f}% object la tiny/small (<32px)** - uu tien kien truc "
      f"giu do phan giai cao (FPN/HRNet), anchor/label-assignment cho object nho (vd RFLA).")
    A(f"4. Co **{_fmt(t['n_background'])} anh background** - giup giam false positive khi train.")
    A("5. File du lieu kem theo: `summary.json`, `instances.csv`, `images.csv` de phan tich sau.\n")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  -> {out_path.name}")
