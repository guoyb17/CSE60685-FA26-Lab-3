"""Download and unpack Fashion-MNIST; reuse existing data files."""
import argparse
import gzip
from pathlib import Path
import urllib.request

from common import ROOT, read_json, run_cli
from data import test_set, training_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    raw = args.data_dir / 'FashionMNIST' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for entry in read_json(ROOT / 'assets' / 'data_manifest.json')['files']:
        compressed = raw / entry['name']
        destination = raw / entry['raw_name']
        if destination.exists():
            continue
        if not compressed.exists():
            print(f'Downloading {entry["name"]}...', flush=True)
            with urllib.request.urlopen(entry['url'], timeout=60) as response:
                content = response.read()
            with compressed.open('xb') as stream:
                stream.write(content)
        with gzip.open(compressed, 'rb') as stream:
            content = stream.read()
        with destination.open('xb') as stream:
            stream.write(content)
    _, train, validation = training_sets(args.data_dir)
    test = test_set(args.data_dir)
    print(f'Ready: {len(train)} training / {len(validation)} validation / {len(test)} test images.')


if __name__ == '__main__':
    run_cli(main)
