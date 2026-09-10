#!/usr/bin/env python3
import sys
import os
import time
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "lexus-x/fishonet-checkpoints-backup"

api = HfApi(token=TOKEN)

def upload(file_path, path_in_repo=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    if path_in_repo is None:
        path_in_repo = os.path.basename(file_path)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"--> Uploading {file_path} ({file_size_mb:.2f} MB) to {REPO_ID}:{path_in_repo}...")
    start_time = time.time()
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="model"
        )
        elapsed = time.time() - start_time
        speed = file_size_mb / elapsed if elapsed > 0 else 0
        print(f"    Uploaded {path_in_repo} in {elapsed:.1f}s ({speed:.2f} MB/s)")
        return True
    except Exception as e:
        print(f"    Failed to upload {file_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            upload(f)
    else:
        print("Usage: upload_to_hf.py <file1> [file2 ...]")
