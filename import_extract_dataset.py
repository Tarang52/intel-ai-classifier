import os
import subprocess
from zipfile import ZipFile
import glob

# Set up your dataset directory
DATA_DIR = "datasets/mechanical_parts"
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset slug from Kaggle URL
KAGGLE_DATASET = "manikantanrnair/images-of-mechanical-parts-boltnut-washerpin"

# 🔽 Step 1: Download the dataset using Kaggle CLI
print("📦 Downloading dataset...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", KAGGLE_DATASET,
    "-p", DATA_DIR,
    "--force"  # Overwrite if already present
], check=True)

# 🔍 Step 2: Find the downloaded ZIP file
zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip"))
if not zip_files:
    raise FileNotFoundError("❌ No zip file found in the dataset directory.")

ZIP_PATH = zip_files[0]
print(f"✅ Found zip file: {os.path.basename(ZIP_PATH)}")

# 📂 Step 3: Extract the ZIP file
print("📂 Extracting...")
with ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print(f"🎉 Done! Dataset is ready at: {DATA_DIR}")
