import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── CONFIG ─────────────────────────────────────────────────
RAW_DIR   = Path("datasets/mechanical_parts/blnw-images-224")
OUT_DIR   = Path("datasets/mechanical_parts_split")
VAL_SPLIT = 0.20   # 20% for val
SEED      = 42

# ─── MAKE FOLDERS ────────────────────────────────────────────
for split in ("train", "val"):
    for cls in RAW_DIR.iterdir():
        if cls.is_dir():
            (OUT_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

# ─── SPLIT & COPY ───────────────────────────────────────────
for cls_dir in RAW_DIR.iterdir():
    if not cls_dir.is_dir(): 
        continue
    images = list(cls_dir.glob("*.*"))  # all files in class folder
    train_files, val_files = train_test_split(
        images, test_size=VAL_SPLIT, random_state=SEED, shuffle=True
    )
    # copy into train/
    for src in train_files:
        dst = OUT_DIR / "train" / cls_dir.name / src.name
        shutil.copy2(src, dst)
    # copy into val/
    for src in val_files:
        dst = OUT_DIR / "val" / cls_dir.name / src.name
        shutil.copy2(src, dst)

print(f"✅ Done! Split data in `{OUT_DIR}` (train/{len(train_files)} + val/{len(val_files)} per class)")
