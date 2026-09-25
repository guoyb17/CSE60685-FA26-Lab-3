"""Create the three required independent branches from one source model."""
import argparse
from pathlib import Path
import torch

from common import (ROOT, evaluate_loader, load_checkpoint, new_directory, run_cli,
                    parameter_count, save_checkpoint, setup_cpu, write_json)
from data import loader, training_sets
from models import MODEL_NAMES
from pruning import build_branch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    setup_cpu()
    source, checkpoint, _ = load_checkpoint(args.baseline, stage='source')
    dataset, _, validation = training_sets(args.data_dir)
    probes = torch.cat([torch.zeros(1, 1, 28, 28), torch.rand(2, 1, 28, 28),
                        torch.stack([validation[i][0] for i in range(3)])])
    output = new_directory(args.output)
    for name in MODEL_NAMES:
        model, pruning = build_branch(source, name, probes)
        metrics = evaluate_loader(model, loader(validation))
        metadata = {'model_name': name, 'stage': 'pruned', 'pruning': pruning,
                    'origin': checkpoint['origin'],
                    'training_history': [], 'before_validation': metrics}
        destination = new_directory(output / name)
        save_checkpoint(destination / 'checkpoint.pt', model, metadata, dataset, metrics)
        write_json(destination / 'pruning.json', {'model_name': name, 'pruning': pruning,
                   'parameter_count': parameter_count(model), 'before_validation': metrics,
                   'origin': metadata['origin']})
        print(f'{name}: {parameter_count(model):,} parameters; '
              f'validation before fine-tuning {metrics["accuracy_pct"]:.2f}%', flush=True)


if __name__ == '__main__':
    run_cli(main)
