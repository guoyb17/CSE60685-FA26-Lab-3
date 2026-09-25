"""Verify a deployment checkpoint and evaluate all 10,000 test images."""
import argparse
from pathlib import Path

from common import (ROOT, environment, evaluate_loader,
                    load_checkpoint, new_directory, run_cli, setup_cpu,
                    verify_reload, write_json)
from data import loader, test_set, training_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--verify-only', action='store_true',
                        help='Check reload and validation without evaluating the test set.')
    args = parser.parse_args()
    setup_cpu()
    model, checkpoint, path = load_checkpoint(args.checkpoint, stage='deployment')
    dataset, _, validation = training_sets(args.data_dir)
    metrics, error = verify_reload(model, checkpoint, dataset, validation)
    output = new_directory(args.output)
    result = {'model_name': checkpoint['model_name'], 'checkpoint': str(path),
              'reload_verified': True, 'reload_max_abs_error': error,
              'validation_accuracy_pct': metrics['accuracy_pct'],
              'environment': environment()}
    if args.verify_only:
        write_json(output / 'reload_verification.json', result)
        print(f'PASS reload: {checkpoint["model_name"]}; validation {metrics["accuracy_pct"]:.2f}%')
        return
    final = evaluate_loader(model, loader(test_set(args.data_dir)))
    result.update(test_accuracy_pct=final['accuracy_pct'], test_correct=final['correct'],
                  test_count=final['count'], test_mean_loss=final['mean_loss'])
    write_json(output / 'evaluation.json', result)
    print(f'PASS reload: {checkpoint["model_name"]}; test {final["correct"]}/10000 '
          f'= {final["accuracy_pct"]:.2f}%')


if __name__ == '__main__':
    run_cli(main)
