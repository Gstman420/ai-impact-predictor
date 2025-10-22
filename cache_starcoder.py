import os
import shutil
from pathlib import Path

# Source and destination
source_dir = "D:/ai_fastapi_project/starcoder2-7b"
cache_base = Path.home() / ".cache" / "huggingface" / "hub"
model_cache_dir = cache_base / "models--bigcode--starcoder2-7b"

print("📦 Caching StarCoder2-7B...")
print(f"Source: {source_dir}")
print(f"Destination: {model_cache_dir}\n")

# Create cache structure
snapshots_dir = model_cache_dir / "snapshots" / "main"
snapshots_dir.mkdir(parents=True, exist_ok=True)

print("Copying files (this will take 5-10 minutes for 14GB)...")

# Copy all files
file_count = 0
for item in Path(source_dir).iterdir():
    if item.is_file():
        dest_file = snapshots_dir / item.name
        print(f"  Copying: {item.name}")
        shutil.copy2(item, dest_file)
        file_count += 1

print(f"\n✅ Copied {file_count} files!")
print("🎉 StarCoder2-7B cached successfully!")