"""Read the three models' results and write the report table."""
import argparse
import csv
from pathlib import Path

from common import new_directory, read_json, run_cli, write_json
from models import MODEL_NAMES


def collect(results):
    rows = []
    for name in MODEL_NAMES:
        folder = results / name
        evaluation = read_json(folder / 'test/evaluation.json')
        timing = read_json(folder / 'timing/summary.json')
        training = read_json(folder / 'training_summary.json')
        rows.append({'model': name, 'parameters': training['parameter_count'],
                     'validation_before_ft_pct': training['validation_before_ft_pct'],
                     'validation_after_ft_pct': training['validation_after_ft_pct'],
                     'test_accuracy_pct': evaluation['test_accuracy_pct'],
                     'median_ms': timing['inference']['median_ms'],
                     'p95_ms': timing['inference']['p95_ms']})
    for row in rows:
        row['speedup_vs_control'] = rows[0]['median_ms'] / row['median_ms']
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
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
