"""
Keep only AI-TOD test set images and clean up train/val images to save disk space.
"""
import shutil
from pathlib import Path

DATA_ROOT = Path(r"D:\paper_a_data\AI-TOD-v2")

def clean_train_val():
    print(f"=== Scanning {DATA_ROOT} to retain ONLY test images ===")
    
    # Check for unzipped AI-TOD structure
    for p in DATA_ROOT.rglob("train"):
        if p.is_dir() and "images" in str(p):
            print(f"Removing training images folder: {p}")
            shutil.rmtree(p, ignore_errors=True)
            
    for p in DATA_ROOT.rglob("val"):
        if p.is_dir() and "images" in str(p):
            print(f"Removing validation images folder: {p}")
            shutil.rmtree(p, ignore_errors=True)

    # Count test images
    test_imgs = list(DATA_ROOT.rglob("test/*.png")) + list(DATA_ROOT.rglob("test/*.jpg"))
    print(f"\n[DONE] Retained ONLY test images: {len(test_imgs)} test images found!")

if __name__ == "__main__":
    clean_train_val()
