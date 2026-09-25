"""Inspect stored parameters, effective nonzero weights and tensor shapes."""
import argparse
import json
from pathlib import Path
from common import load_checkpoint, run_cli, setup_cpu, weight_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    setup_cpu()
    model, checkpoint, _ = load_checkpoint(args.checkpoint)
    print(checkpoint['model_name'], json.dumps(weight_statistics(model), indent=2))


if __name__ == '__main__':
    run_cli(main)
