"""
Downlaod models from hugggingface hub .
"""
from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument('repo_id')
    args = parser.parse_args()

    tk = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code = True)
    model = AutoModel.from_pretrained(args.repo_id, trust_remote_code = True)

if __name__ == '__main__':
    main()