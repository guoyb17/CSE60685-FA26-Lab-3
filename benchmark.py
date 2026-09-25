"""Measure CPU inference only, with a preloaded batch-one input tensor."""
import argparse
import csv
import hashlib
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from common import (ROOT, checkpoint_provenance, course_config, device_snapshot, environment,
                    load_checkpoint, new_directory, run_cli, setup_cpu, write_json)
from data import benchmark_input, check_checkpoint_data


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('Use a positive integer.')
    return value


def measure(model, tensor, warmup=10, runs=100):
    model.eval()
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        for _ in range(runs):
            # TIMING START: the prepared tensor and evaluation mode are already set.
            start = perf_counter_ns()
            model(tensor)
            end = perf_counter_ns()
            # TIMING END: recording, statistics and output are excluded.
            samples.append((end - start) / 1_000_000)
    return samples


def statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Expected positive, finite timing samples.')
    return {'sample_count': int(values.size), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95, method='linear')),
            'mean_ms': float(np.mean(values)), 'min_ms': float(np.min(values)),
            'max_ms': float(np.max(values))}


def main():
    cfg = course_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--threads', type=positive_int, default=cfg['threads'])
    parser.add_argument('--warmup', type=positive_int, default=cfg['warmup'])
    parser.add_argument('--runs', type=positive_int, default=cfg['runs'])
    args = parser.parse_args()
    setup_cpu(args.threads)
    model, checkpoint, path = load_checkpoint(args.checkpoint, stage='deployment')
    check_checkpoint_data(checkpoint)
    tensor, index, label = benchmark_input(args.data_dir)
    output = new_directory(args.output)
    before = device_snapshot()
    samples = measure(model, tensor, args.warmup, args.runs)
    after = device_snapshot()
    with (output / 'raw_timings.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['iteration', 'latency_ms'])
        writer.writerows(enumerate(samples, 1))
    result = {**checkpoint_provenance(checkpoint, path), 'schema_version': 3,
              'runtime': 'PyTorch eager CPU', 'threads': args.threads, 'warmup': args.warmup,
              'runs': args.runs, 'input_shape': list(tensor.shape), 'input_dtype': 'float32',
              'input_index': index, 'input_label': label,
              'input_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
              'inference': statistics(samples), 'environment': environment(),
              'telemetry': {'before': before, 'after': after},
              'timing_scope': 'Synchronous CPU model(x), including Python call overhead; excludes loading, preprocessing, argmax/softmax, printing, checksums and output files.'}
    write_json(output / 'summary.json', result)
    print(f'{checkpoint["model_name"]}: inference median {result["inference"]["median_ms"]:.4f} ms; '
          f'p95 {result["inference"]["p95_ms"]:.4f} ms; {len(samples)} recorded samples')


if __name__ == '__main__':
    run_cli(main)
