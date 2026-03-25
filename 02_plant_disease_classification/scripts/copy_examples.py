"""
Copy 10 example images from the test set into examples/ for the Gradio demo.

Usage:
  python scripts/copy_examples.py

Prerequisites:
  - data/raw/ must be populated (download PlantVillage first:
    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
  - data/processed/test.csv must exist (already committed to repo)

What it does:
  - Reads data/processed/test.csv
  - For each of 10 target classes, picks the first image path alphabetically
  - Copies it to examples/<NN>_<clean_name>.jpg
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
TEST_CSV     = BASE_DIR / "data" / "processed" / "test.csv"
EXAMPLES_DIR = BASE_DIR / "examples"

# ── target classes and output filenames ────────────────────────────────────
# Keys must match the 'label' column in test.csv exactly.
# If a class name does not match, the script will list available names.
TARGETS = {
    "Tomato___Late_blight":                 "01_tomato_late_blight.jpg",
    "Tomato___healthy":                     "02_tomato_healthy.jpg",
    "Apple___Apple_scab":                   "03_apple_scab.jpg",
    "Corn_(maize)___Northern_Leaf_Blight":  "04_corn_nlb.jpg",
    "Potato___Late_blight":                 "05_potato_late_blight.jpg",
    "Grape___Black_rot":                    "06_grape_black_rot.jpg",
    "Peach___Bacterial_spot":               "07_peach_bacterial_spot.jpg",
    "Pepper,_bell___healthy":               "08_pepper_bell_healthy.jpg",
    "Strawberry___Leaf_scorch":             "09_strawberry_leaf_scorch.jpg",
    "Apple___healthy":                      "10_apple_healthy.jpg",
}


def main():
    # ── sanity checks ──────────────────────────────────────────────────────
    if not TEST_CSV.exists():
        print(f"ERROR: test.csv not found at {TEST_CSV}")
        sys.exit(1)

    df = pd.read_csv(TEST_CSV)

    raw_dir = BASE_DIR / "data" / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print("ERROR: data/raw/ is empty.")
        print("Download PlantVillage first:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("Then unzip into data/raw/ so paths match those in test.csv.")
        sys.exit(1)

    EXAMPLES_DIR.mkdir(exist_ok=True)

    # ── validate class names ───────────────────────────────────────────────
    available_classes = sorted(df["label"].unique())
    unknown = [c for c in TARGETS if c not in available_classes]
    if unknown:
        print("WARNING: these class names were not found in test.csv:")
        for c in unknown:
            print(f"  '{c}'")
        print("\nAvailable class names:")
        for c in available_classes:
            print(f"  '{c}'")
        print("\nEdit TARGETS in this script to match exact class names above.")
        sys.exit(1)

    # ── copy images ────────────────────────────────────────────────────────
    print(f"Copying examples to {EXAMPLES_DIR}/\n")
    copied = 0

    for class_name, output_name in TARGETS.items():
        # All rows for this class, sorted by path for determinism
        rows = df[df["label"] == class_name].sort_values("image_path")

        if rows.empty:
            print(f"  SKIP  {output_name}  — no rows for '{class_name}' in test.csv")
            continue

        # Paths in test.csv are relative to project root
        src = BASE_DIR / rows.iloc[0]["image_path"]

        if not src.exists():
            print(f"  MISS  {output_name}  — file not on disk: {src}")
            continue

        dst = EXAMPLES_DIR / output_name
        shutil.copy2(src, dst)
        print(f"  OK    {output_name}  ← {src.name}")
        copied += 1

    print(f"\n{copied}/{len(TARGETS)} images copied to examples/")

    if copied == len(TARGETS):
        print("\nAll done. Now uncomment the  examples=EXAMPLES  line in app.py.")
    else:
        missing = len(TARGETS) - copied
        print(f"\n{missing} image(s) missing — check data/raw/ is fully extracted.")


if __name__ == "__main__":
    main()
