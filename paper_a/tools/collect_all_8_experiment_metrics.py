"""
Comprehensive Metric Collector & Log Parser for All 8 Journal & Benchmark Experiments.
Downloads latest artifacts from all 8 accounts and extracts full metric records.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
RESULTS_DIR = ROOT / "journal/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {"stt": 1, "dataset": "AI-TOD-v2", "slug": "tod-aitod-baseline-s42", "acc": "amongus1504", "cred": "kaggle.json", "name": "Faster R-CNN Baseline (8 classes)"},
    {"stt": 2, "dataset": "AI-TOD-v2", "slug": "tod-aitod-hwiou-sig8-s42", "acc": "qnhat1504", "cred": "kaggle (3).json", "name": "H-WIoU (sigma_0 = 8.0 px, 8 classes)"},
    {"stt": 3, "dataset": "AI-TOD-v2", "slug": "tod-aitod-hwiou-sig6-s42", "acc": "quangnhtng", "cred": "kaggle (6).json", "name": "H-WIoU (sigma_0 = 6.0 px, 8 classes)"},
    {"stt": 4, "dataset": "AI-TOD-v2", "slug": "tod-aitod-hwiou-sig10-s42", "acc": "phuc1806", "cred": "kaggle (12).json", "name": "H-WIoU (sigma_0 = 10.0 px, 8 classes)"},
    {"stt": 5, "dataset": "TinyPerson", "slug": "tod-tp-ablation-pure-w2-s42", "acc": "thyngluthy", "cred": "kaggle (4).json", "name": "Ablation Pure W2 (gamma = 0)"},
    {"stt": 6, "dataset": "TinyPerson", "slug": "tod-tp-ablation-pure-iou-s42", "acc": "hienquang06", "cred": "kaggle (5).json", "name": "Ablation Pure IoU (gamma = 1)"},
    {"stt": 7, "dataset": "TinyPerson", "slug": "tod-tp-ablation-static-half-s42", "acc": "dipphmngc", "cred": "kaggle (11).json", "name": "Ablation Static Blend (gamma = 0.5)"},
    {"stt": 8, "dataset": "TinyPerson", "slug": "tod-tp-ablation-exp-form-s42", "acc": "hngngnguynvn", "cred": "kaggle (1).json", "name": "Ablation Exponential gamma_exp"},
]


def fetch_and_parse(exp: dict):
    acc = exp["acc"]
    slug = exp["slug"]
    name = exp["name"]
    cred_file = exp["cred"]

    creds = json.loads((CREDS_DIR / cred_file).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    ref = f"{acc}/{slug}"
    out_dir = RESULTS_DIR / f"{acc}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download output
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(out_dir)], env=env, capture_output=True)

    # Parse logs
    logs = list(out_dir.rglob("*.log"))
    metric_entries = []
    error_entries = []
    
    for l_path in logs:
        text = l_path.read_text(encoding="utf-8", errors="replace")
        try:
            items = json.loads(text)
            for it in items:
                d = it.get("data", "")
                if "Epoch " in d and ("AP=" in d or "mAP" in d or "AP50" in d):
                    metric_entries.append(d.strip())
                elif "Traceback" in d or "Error:" in d:
                    error_entries.append(d.strip())
        except Exception:
            for line in text.splitlines():
                if "Epoch " in line and ("AP=" in line or "mAP" in line or "AP50" in line):
                    metric_entries.append(line.strip())
                elif "Traceback" in line or "Error:" in line:
                    error_entries.append(line.strip())

    return {
        "exp": exp,
        "metrics": metric_entries,
        "errors": error_entries,
        "files": [f.name for f in out_dir.iterdir()]
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 115)
    print("           COMPREHENSIVE MULTI-ACCOUNT JOURNAL & ABLATION EXPERIMENT EVALUATION & AUDIT           ")
    print("=" * 115)

    for exp in EXPERIMENTS:
        res = fetch_and_parse(exp)
        stt = exp["stt"]
        dset = exp["dataset"]
        acc = exp["acc"]
        slug = exp["slug"]
        name = exp["name"]
        metrics = res["metrics"]
        errors = res["errors"]

        print(f"\n[{stt}/8] [{dset}] {name} ({acc}/{slug})")
        if metrics:
            print(f"  --> Status: SUCCEEDED ({len(metrics)} validation checkpoints recorded)")
            for m in metrics[-3:]:
                print(f"      * {m}")
        elif errors:
            print(f"  --> Status: ERROR / TERMINATED")
            for e in errors[:3]:
                print(f"      ! {e[:120]}")
        else:
            print(f"  --> Status: RUNNING / INITIALIZING (Files downloaded: {len(res['files'])})")

    print("\n" + "=" * 115)


if __name__ == "__main__":
    main()
