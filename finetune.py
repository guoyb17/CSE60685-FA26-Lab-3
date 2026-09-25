"""Continue each branch for the same three epochs and save a clean deployment model."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from time import perf_counter
import torch

from common import (ROOT, course_config, device_snapshot, environment, evaluate_loader,
                    load_checkpoint, new_directory, run_cli, save_checkpoint, setup_cpu,
                    sha256, verify_reload,
                    weight_statistics, write_json)
from data import check_checkpoint_data, loader, training_sets


def train_epoch(model, batches, optimizer):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    # TIMING START: one complete training loop, including fixed batch statistics.
    started = perf_counter()
    for inputs, targets in batches:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item() * targets.numel()
        correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        count += targets.numel()
    seconds = perf_counter() - started
    # TIMING END: validation, printing and persistence happen in main.
    if count != course_config()['train_size'] or not math.isfinite(loss_sum):
        raise ValueError('Expected a complete epoch with finite training loss.')
    return {'train_loss': loss_sum / count, 'train_accuracy_pct': 100.0 * correct / count,
            'train_epoch_s': seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    cfg = course_config()
    setup_cpu(seed=cfg['seed'])
    model, checkpoint, path = load_checkpoint(args.checkpoint, stage='pruned')
    check_checkpoint_data(checkpoint)
    dataset, training, validation = training_sets(args.data_dir)
    initial, _ = verify_reload(model, checkpoint, dataset, validation)
    output = new_directory(args.output)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    batches = loader(training, training=True)
    history = []
    before_telemetry = device_snapshot()
    for epoch in range(1, cfg['epochs'] + 1):
        row = {'epoch': epoch, **train_epoch(model, batches, optimizer)}
        metrics = evaluate_loader(model, loader(validation))
        row.update(validation_accuracy_pct=metrics['accuracy_pct'], validation_loss=metrics['mean_loss'])
        history.append(row)
        print(f'{checkpoint["model_name"]} epoch {epoch}/{cfg["epochs"]}: loss {row["train_loss"]:.4f}; '
              f'validation {metrics["accuracy_pct"]:.2f}%; train {row["train_epoch_s"]:.2f} s', flush=True)
    metadata = {key: checkpoint[key] for key in ('model_name', 'origin', 'pruning',
                'split_sha256', 'data_manifest_sha256', 'before_validation')}
    metadata.update(stage='deployment', training_history=history,
                    parent_checkpoint_sha256=sha256(path))
    save_checkpoint(output / 'checkpoint.pt', model, metadata, dataset, metrics)
    with (output / 'training.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(output / 'config.json', {'architecture': model.config, 'course_config': cfg,
               'origin': metadata['origin'], 'pruning': metadata['pruning']})
    write_json(output / 'training_summary.json', {'model_name': checkpoint['model_name'],
               'checkpoint_sha256': sha256(output / 'checkpoint.pt'),
               'validation_before_ft_pct': initial['accuracy_pct'],
               'validation_after_ft_pct': metrics['accuracy_pct'],
               'mean_train_epoch_s': mean(row['train_epoch_s'] for row in history),
               'weights': weight_statistics(model), 'environment': environment(),
               'telemetry': {'before': before_telemetry, 'after': device_snapshot()}})
    print('Saved last-epoch deployment checkpoint.')


if __name__ == '__main__':
    run_cli(main)
