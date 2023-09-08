"""
Download huggingface repository into specified path and do not cache to default dir.
Document:
- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.snapshot_download
"""

from huggingface_hub import hf_hub_download, snapshot_download
import joblib
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('repo_id', help = 'Huggingface repository name, e.g., THUDM/chatglm2-6b')
    parser.add_argument('local_dir', help = 'local dir to save the model')
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id, 
        local_dir= args.local_dir, 
        local_dir_use_symlinks = False
    )

if __name__ == '__main__':
    main()