# CSE 60685 Fall 2026 Lab 3

Pruning and Fine-Tuning a CNN on Raspberry Pi.

Follow the Lab 3 Tutorial and Assignment.
All code is complete; no implementation exercises are required.
The camera is not needed.

## Prepare the project

After cloning this repo, continue in a Pi terminal:

```bash
cd ~/CSE60685-FA26-Lab-3
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
python prepare_data.py
```

Use a separate environment.
In a new terminal, return to this folder and run `source env/bin/activate`.

## Import A of Lab 2

```bash
python import_baseline.py --checkpoint ../CSE60685-FA26-Lab-2/results/baseline/checkpoint.pt --output results/source
```

Change the input path if necessary.
Use A's last checkpoint from the required five-epoch run, with channels 6/16 and hidden widths 120/84.
The import verifies architecture, settings, data split, saved logits and validation accuracy.

If your checkpoint is unavailable or fails verification, check it first.
You may use `--checkpoint assets/fallback_baseline.pt` instead;
identify the fallback in your report.

Keep the published, disjoint, class-balanced split: 12,000 training images and 2,000 validation images, plus all 10,000 official test images for final evaluation.
Inputs are 28 × 28 grayscale float32 tensors in [0,1], with no extra normalization or augmentation.

## Read and check the supplied pruning code

`pruning_ops.py` contains complete implementations of channel selection and weight transfer.
Read how the code retains Conv2 filters, their biases and the corresponding FC1 input columns.
Every pooled channel has 16 features, so its FC1 columns must be selected together.

```bash
python check_pruning.py
```

The expected results are `PASS s25` and `PASS s50`.
The check compares the compact model with the full model whose removed channels are zeroed after the second pooling layer, rather than with the source.

| Model | Conv2 channels | FC1 inputs | Parameters |
| --- | ---: | ---: | ---: |
| Control | 16 | 256 | 44,426 |
| S25 | 12 | 192 | 36,142 |
| S50 | 8 | 128 | 27,858 |

S25/S50 indicate the fraction of Conv2 output channels removed.

## Fine-tune and evaluate

```bash
python prune_models.py --baseline results/source/checkpoint.pt --output results/pruned

for model in control s25 s50; do
    python finetune.py --checkpoint results/pruned/$model/checkpoint.pt --output results/$model
done

for model in control s25 s50; do
    python evaluate.py --checkpoint results/$model/checkpoint.pt --output results/$model/test
done
```

All three models start independently from the source.
Each receives three additional epochs: batch size 64, a new Adam optimizer at 0.0001, seed 42, one intra-op thread, one inter-op thread and zero loader workers.
Keep the supplied settings and use the last epoch.
Note: the control also continues training.

`training.csv` records the learning progress.
`training_summary.json` records validation accuracy before and after fine-tuning.
For the control, before means source accuracy;
for S25/S50, it means accuracy immediately after pruning.

Evaluation reloads the saved architecture and weights, verifies saved logits and validation accuracy, then evaluates the complete test set.
Look for `PASS reload`.
The final results are in `test/evaluation.json`.

## Benchmark and report

Finish training and evaluation first.
Keep power, cooling, software and other activity unchanged.
Let the Pi reach comparable starting temperatures and run one benchmark at a time.

```bash
for model in control s25 s50; do
    python benchmark.py --checkpoint results/$model/checkpoint.pt --threads 1 --warmup 10 --runs 100 --output results/$model/timing
done

python summarize.py --results results --output results/comparison
```

Each model uses the same preloaded validation image, shape [1,1,28,28], float32.
The benchmark uses evaluation and inference modes.
Timing includes the synchronous CPU model call and Python overhead;
loading, preprocessing, argmax/softmax, printing and saving are excluded.

Each raw timing CSV has exactly 100 measurement rows plus a header, excluding the 10 warm-ups.
Report the median in milliseconds.
The summary verifies checkpoint pairing, source, settings, test count and raw timing statistics, then writes `comparison.csv`, `comparison.json` and `report_table.md` for the three required models.

## Troubleshooting and file guide

- `models.py`: completed network and fixed required model configurations.
- `pruning_ops.py`: complete L1 channel selection and weight-copy functions.
- `pruning.py`: model construction and consistency checks.
- `common.py`, `data.py`: shared CPU, checkpoint and dataset helpers following Lab 2.
- `assets/split.json`, `assets/data_manifest.json`: unchanged Lab 2 data assets.
- `assets/fallback_baseline.pt`: supplied five-epoch Lab 2 A, available when a personal baseline cannot be used.

A source error usually means the wrong Lab 2 model, checkpoint path, training recipe or split.
If the pruning check fails in a fresh copy, preserve its output and contact the TA.

Programs refuse existing output directories.
For a complete repeat, use a new root such as `results_repeat1` consistently in all paths and pass it as `--results` to the summary.
Do not mix results from different runs.

Data preparation checks existing files.
If a checksum fails, preserve them for diagnosis and prepare a new directory.
Pass the same `--data-dir` to every later command.

A separate reload-only diagnostic is available:
```bash
python evaluate.py --checkpoint results/control/checkpoint.pt --verify-only --output results/control/reload_check
```
The normal evaluation already performs this check, so a separate diagnostic run is not required.
Benchmark the final deployment checkpoint, not the intermediate file under `results/pruned`.
