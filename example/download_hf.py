"""
Download huggingface repository into specified path and do not cache to default dir.
Document:
- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.snapshot_download
"""

from huggingface_hub import hf_hub_download, snapshot_download
import joblib

REPO_ID = "THUDM/chatglm2-6b"
LOCAL_NAME = ""
model = snapshot_download(
    repo_id=REPO_ID, 
    local_dir=f'/data1/llm/{LOCAL_NAME}', 
    local_dir_use_symlinks = False
)