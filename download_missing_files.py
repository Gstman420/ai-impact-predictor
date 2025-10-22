from huggingface_hub import hf_hub_download
import os

model_id = "Qwen/Qwen3-Embedding-8B"
local_dir = "D:/ai_fastapi_project/Qwen3-Embedding-8B"

# Missing files
missing_files = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors", 
    "model-00003-of-00004.safetensors"
]

print("Downloading missing Qwen model files...")
print("This will take 10-20 minutes (large files!)\n")

for filename in missing_files:
    print(f"Downloading: {filename}")
    try:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {filename}\n")
    except Exception as e:
        print(f"Error downloading {filename}: {e}\n")

print("Download complete!")