from huggingface_hub import snapshot_download

model_id = "facebook/bart-large-cnn"  # Popular, reliable summarizer
local_dir = "D:/ai_fastapi_project/bart-summarizer"

print("📥 Downloading BART Summarizer (better model)...")
print("⚠️ Size: ~1.6GB\n")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("\n✅ BART Summarizer downloaded!")
except Exception as e:
    print(f"❌ Error: {e}")