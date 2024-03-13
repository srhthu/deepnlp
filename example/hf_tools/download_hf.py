"""
Download huggingface repository into specified path and do not cache to default dir.
Document:
- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/file_download#huggingface_hub.snapshot_download
"""

from huggingface_hub import hf_hub_download, snapshot_download
import joblib
from argparse import ArgumentParser


def exec_download(repo_id, cache_dir = None, local_dir = None):
    kws = {}
    if cache_dir:
        kws.update({'cache_dir': cache_dir})
    if local_dir:
        kws.update({'local_dir': local_dir, 'local_dir_use_symlinks': False})

    print(f'Start downloading {repo_id}, kws: {kws}')
    ss_dir = snapshot_download(
        repo_id=repo_id, 
        token = True,
        **kws
    )
    print(f'Download finished: {ss_dir}')

def main():
    parser = ArgumentParser()
    parser.add_argument('repo_id', help = 'Huggingface repository name, e.g., THUDM/chatglm2-6b')
    parser.add_argument('--cache_dir', help = 'If not specified, use the default cache dir.')
    parser.add_argument('--local_dir', help = 'local dir to save the model')
    args = parser.parse_args()

    exec_download(args.repo_id, 
                  cache_dir=args.cache_dir, 
                  local_dir = args.local_dir)

if __name__ == '__main__':
    main()