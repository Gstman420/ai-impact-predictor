 

import os
import shutil
from pathlib import Path

# Source directory
SOURCE_DIR = Path("D:/ai_fastapi_project")

# Target cache directory
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

# Model mappings (folder name -> proper model name)
MODEL_MAPPING = {
    "RequirementClassifier": "models--rajinikarcg--RequirementClassifier",
    "Qwen3-Embedding-8B": "models--Qwen--Qwen3-Embedding-8B",
    "M365_h2_Text_Processing_and_Summarization": "models--marklicata--M365_h2_Text_Processing_and_Summarization",
    "roberta-financial-news-impact-analysis": "models--nusret35--roberta-financial-news-impact-analysis"
}

print("=" * 70)
print("🔧 SETTING UP HUGGINGFACE MODEL CACHE")
print("=" * 70)
print(f"Source: {SOURCE_DIR}")
print(f"Target: {CACHE_DIR}")
print()

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"✅ Cache directory ready: {CACHE_DIR}\n")

# Process each model
for source_name, target_name in MODEL_MAPPING.items():
    source_path = SOURCE_DIR / source_name
    
    if not source_path.exists():
        print(f"⚠️  SKIPPING: {source_name} (not found)")
        continue
    
    print(f"📦 Processing: {source_name}")
    
    # Create target structure
    target_base = CACHE_DIR / target_name
    
    # Create a snapshot folder (using a dummy hash)
    snapshot_hash = "main"
    snapshot_dir = target_base / "snapshots" / snapshot_hash
    
    # Create refs folder
    refs_dir = target_base / "refs"
    
    # Create directories
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  - Created structure: {target_base}")
    
    # Copy all files from source to snapshot
    copied_files = 0
    for file in source_path.glob("*"):
        if file.is_file():
            target_file = snapshot_dir / file.name
            shutil.copy2(file, target_file)
            copied_files += 1
    
    print(f"  - Copied {copied_files} files")
    
    # Create main ref file
    main_ref = refs_dir / "main"
    main_ref.write_text(snapshot_hash)
    
    print(f"  ✅ {source_name} -> {target_name}\n")

print("=" * 70)
print("🎉 SETUP COMPLETE!")
print("=" * 70)
print(f"\n📍 Models installed at: {CACHE_DIR}")
print("\n🧪 Next step: Run test to verify models work!")
EOF