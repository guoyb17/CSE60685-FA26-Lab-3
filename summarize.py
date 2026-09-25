"""Validate matching final results and write the three-row report table."""
import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark import statistics
from common import (course_config, load_checkpoint, new_directory, read_json,
                    run_cli, setup_cpu, sha256, write_json)
from data import check_checkpoint_data, split_spec
from models import MODEL_NAMES


def collect(results):
    cfg = course_config()
    rows, sources, inputs, environments = [], set(), set(), set()
    for name in MODEL_NAMES:
        folder = results / name
        _, checkpoint, path = load_checkpoint(folder / 'checkpoint.pt', stage='deployment')
        check_checkpoint_data(checkpoint)
        if checkpoint['model_name'] != name:
            raise ValueError(f'{folder}: wrong model.')
        digest = sha256(path)
        evaluation = read_json(folder / 'test/evaluation.json')
        timing = read_json(folder / 'timing/summary.json')
        training = read_json(folder / 'training_summary.json')
        for result in (evaluation, timing, training):
            if result['checkpoint_sha256'] != digest or result['model_name'] != name:
                raise ValueError(f'{name}: results refer to different checkpoints.')
        if evaluation['test_count'] != 10000 or not evaluation['reload_verified']:
            raise ValueError(f'{name}: evaluate the complete test set after verifying reload.')
        if abs(evaluation['test_accuracy_pct'] - evaluation['test_correct'] / 100) > 1e-9:
            raise ValueError(f'{name}: inconsistent test accuracy.')
        final_val = checkpoint['verification']['validation_accuracy_pct']
        if training['validation_before_ft_pct'] != checkpoint['before_validation']['accuracy_pct']:
            raise ValueError(f'{name}: inconsistent validation accuracy before fine-tuning.')
        if training['validation_after_ft_pct'] != final_val or evaluation['validation_accuracy_pct'] != final_val:
            raise ValueError(f'{name}: inconsistent final validation accuracy.')
        expected = {'threads': 1, 'warmup': 10, 'runs': 100, 'input_shape': [1, 1, 28, 28],
                    'input_dtype': 'float32', 'runtime': 'PyTorch eager CPU',
                    'input_index': split_spec()['benchmark_index']}
        if any(timing[key] != value for key, value in expected.items()):
            raise ValueError(f'{name}: use the required one-thread batch-one timing settings.')
        for result in (evaluation, timing):
            if result['origin'] != checkpoint['origin'] or result['course_config'] != cfg:
                raise ValueError(f'{name}: source or course settings do not match.')
        with (folder / 'timing/raw_timings.csv').open(newline='', encoding='utf-8') as stream:
            raw = list(csv.DictReader(stream))
        if [int(row['iteration']) for row in raw] != list(range(1, 101)):
            raise ValueError(f'{name}: expected exactly 100 raw timing rows.')
        actual = statistics([float(row['latency_ms']) for row in raw])
        if not all(np.isclose(value, timing['inference'][key], rtol=1e-9, atol=1e-12)
                   for key, value in actual.items()):
            raise ValueError(f'{name}: timing summary differs from the raw measurements.')
        env = timing['environment']
        if env['threads'] != 1 or env['interop_threads'] != 1 or env['execution_device'] != 'cpu':
            raise ValueError(f'{name}: benchmark CPU settings do not match.')
        environments.add(tuple(env.get(key) for key in ('system', 'machine', 'pi_model', 'python', 'torch', 'numpy')))
        sources.add(checkpoint['origin']['imported_checkpoint_sha256'])
        inputs.add(timing['input_sha256'])
        rows.append({'model': name, 'parameters': checkpoint['parameter_count'],
                     'validation_before_ft_pct': training['validation_before_ft_pct'],
                     'validation_after_ft_pct': final_val,
                     'test_accuracy_pct': evaluation['test_accuracy_pct'],
                     'median_ms': timing['inference']['median_ms'],
                     'p95_ms': timing['inference']['p95_ms']})
    if len(sources) != 1 or len(inputs) != 1 or len(environments) != 1:
        raise ValueError('Use one imported source, benchmark image and device/runtime for all rows.')
    for row in rows:
        row['speedup_vs_control'] = rows[0]['median_ms'] / row['median_ms']
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    setup_cpu()
    rows = collect(args.results)
    output = new_directory(args.output)
    with (output / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'comparison.json', rows)
    text = ['| Model | Parameters | Validation before -> after FT (%) | Test (%) | Median (ms) |',
            '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        text.append(f'| {row["model"]} | {row["parameters"]:,} | '
                    f'{row["validation_before_ft_pct"]:.2f} -> {row["validation_after_ft_pct"]:.2f} | '
                    f'{row["test_accuracy_pct"]:.2f} | {row["median_ms"]:.4f} |')
    (output / 'report_table.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text))


if __name__ == '__main__':
    run_cli(main)
