"""Download and verify the official dataset; preserve any existing files."""
import argparse
import gzip
import hashlib
from pathlib import Path
import tempfile
import urllib.request

from common import ROOT, read_json, run_cli, sha256
from data import training_sets, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    raw = args.data_dir / 'FashionMNIST' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for entry in read_json(ROOT / 'assets' / 'data_manifest.json')['files']:
        compressed = raw / entry['name']
        if compressed.exists():
            if sha256(compressed) != entry['sha256']:
                raise ValueError(f'Existing download is corrupt: {compressed}. Use a new --data-dir.')
        else:
            print(f'Downloading {entry["name"]}...', flush=True)
            with urllib.request.urlopen(entry['url'], timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError(f'Download checksum mismatch for {entry["name"]}; file not installed.')
            with compressed.open('xb') as stream:
                stream.write(content)
        destination = raw / entry['raw_name']
        if not destination.exists():
            with gzip.open(compressed, 'rb') as stream:
                content = stream.read()
            if hashlib.sha256(content).hexdigest() != entry['raw_sha256']:
                raise ValueError(f'Decoded checksum mismatch for {entry["name"]}.')
            with destination.open('xb') as stream:
                stream.write(content)
    verify_files(args.data_dir)
    _, train, validation = training_sets(args.data_dir, verify=False)
    print(f'PASS: {len(train)} training / {len(validation)} validation; official test set verified.')


if __name__ == '__main__':
    run_cli(main)
