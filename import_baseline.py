"""Load Lab 2 A and save a copy as the Lab 3 source model."""
import argparse
from pathlib import Path
import torch

from common import (ROOT, course_config, environment, new_directory, parameter_count,
                    run_cli, save_checkpoint, setup_cpu, verify_reload, write_json)
from data import training_sets
from models import BASELINE, LeNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    setup_cpu()
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if original.get('model_name') != 'baseline':
        raise ValueError('Use the Lab 2 A baseline checkpoint, not B, C or an exported ONNX model.')
    if original.get('architecture') != BASELINE or original.get('parameter_count') != 44426:
        raise ValueError('The source must have channels 6/16 and hidden widths 120/84.')
    required_settings = {'dataset': 'Fashion-MNIST', 'train_size': 12000, 'validation_size': 2000,
                         'epochs': 5, 'batch_size': 64, 'optimizer': 'Adam', 'learning_rate': 0.001,
                         'seed': 42, 'preprocessing': course_config()['preprocessing']}
    if any(original['course_config'].get(key) != value for key, value in required_settings.items()):
        raise ValueError('The source does not use the required Lab 2 baseline recipe.')
    if [row['epoch'] for row in original['training_history']] != list(range(1, 6)):
        raise ValueError('Use the last checkpoint from the complete five-epoch Lab 2 run.')
    model = LeNet(BASELINE).cpu()
    model.load_state_dict(original['state_dict'], strict=True)
    dataset, _, validation = training_sets(args.data_dir)
    metrics, error = verify_reload(model, original, dataset, validation)
    origin = {'lab2_checkpoint': str(args.checkpoint.expanduser().resolve()), 'lab2_epochs': 5,
              'used_fallback': args.checkpoint.expanduser().resolve() ==
                               (ROOT / 'assets/fallback_baseline.pt').resolve()}
    metadata = {'model_name': 'source', 'stage': 'source', 'origin': origin,
                'pruning': {'method': 'none', 'scope': 'Lab 2 source'},
                'training_history': [], 'before_validation': metrics}
    output = new_directory(args.output)
    save_checkpoint(output / 'checkpoint.pt', model, metadata, dataset, metrics)
    write_json(output / 'source.json', {'origin': origin, 'architecture': BASELINE,
               'parameter_count': parameter_count(model), 'validation': metrics,
               'reload_max_abs_error': error, 'environment': environment()})
    print(f'PASS source: 44,426 parameters; validation {metrics["accuracy_pct"]:.2f}%; fallback={origin["used_fallback"]}')


if __name__ == '__main__':
    run_cli(main)
