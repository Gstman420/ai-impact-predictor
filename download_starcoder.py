from huggingface_hub import snapshot_download
import os

model_id = "bigcode/starcoder2-7b"
local_dir = "D:/ai_fastapi_project/starcoder2-7b"

print("📥 Downloading StarCoder2-7B (Code Analyzer)...")
print("⚠️ Size: ~7GB - This will take 15-20 minutes\n")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("\n🎉 StarCoder2-7B downloaded successfully!")
    print(f"📍 Location: {local_dir}")
except Exception as e:
    print(f"❌ Error: {e}")