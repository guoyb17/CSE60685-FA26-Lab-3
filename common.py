"""CPU setup, checkpoint validation and result helpers, following Lab 2."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def course_config():
    return read_json(ROOT / 'course_config.json')


def setup_cpu(threads=1, seed=42):
    if type(threads) is not int or threads < 1:
        raise ValueError('Thread count must be a positive integer.')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_directory(path):
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Use a new --output directory.')
    destination.mkdir(parents=True)
    return destination


def device_snapshot():
    def read_text(path):
        try:
            return Path(path).read_text().strip('\0\n ')
        except OSError:
            return None
    temperature = read_text('/sys/class/thermal/thermal_zone0/temp')
    frequency = read_text('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    throttled = None
    try:
        result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                text=True, timeout=3, check=True)
        throttled = result.stdout.strip().split('=', 1)[-1]
    except (OSError, subprocess.SubprocessError):
        pass
    return {'temperature_c': float(temperature) / 1000 if temperature else None,
            'cpu_frequency_khz': int(frequency) if frequency else None,
            'throttled_hex': throttled}


def environment():
    try:
        pi_model = Path('/proc/device-tree/model').read_text().strip('\0\n ')
    except OSError:
        pi_model = None
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'system': platform.platform(),
            'machine': platform.machine(), 'pi_model': pi_model,
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'execution_device': 'cpu'}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def weight_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]


def weight_statistics(model):
    layers = []
    for name, module in weight_modules(model):
        weight = module.weight.detach()
        total = weight.numel()
        nonzero = int(torch.count_nonzero(weight).item())
        layers.append({'layer': name, 'weight_shape': list(weight.shape),
                       'weight_elements': total, 'nonzero_weights': nonzero})
    total = sum(row['weight_elements'] for row in layers)
    nonzero = sum(row['nonzero_weights'] for row in layers)
    return {'parameter_count': parameter_count(model), 'weight_elements': total,
            'nonzero_weights': nonzero, 'weight_sparsity_pct': 100.0 * (total - nonzero) / total,
            'layers': layers}


def evaluate_loader(model, batches):
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for inputs, targets in batches:
            logits = model(inputs)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.numel()
    if count == 0 or not math.isfinite(loss_sum):
        raise ValueError('Evaluation needs nonempty data and finite logits.')
    return {'correct': correct, 'count': count, 'accuracy_pct': 100.0 * correct / count,
            'mean_loss': loss_sum / count}


def verification_record(model, dataset, validation_metrics):
    from data import split_spec
    index = split_spec()['benchmark_index']
    model.eval()
    with torch.inference_mode():
        logits = model(dataset[index][0].unsqueeze(0)).tolist()
    return {'index': index, 'logits': logits,
            'validation_accuracy_pct': validation_metrics['accuracy_pct']}


def verify_reload(model, checkpoint, dataset, validation):
    from data import loader, split_spec
    record = checkpoint['verification']
    if record['index'] != split_spec()['benchmark_index']:
        raise ValueError('Saved reload probe does not match the course validation image.')
    model.eval()
    with torch.inference_mode():
        actual = model(dataset[record['index']][0].unsqueeze(0))
    expected = torch.tensor(record['logits'], dtype=torch.float32)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise ValueError('Reloaded logits differ from the saved verification output.')
    metrics = evaluate_loader(model, loader(validation))
    if abs(metrics['accuracy_pct'] - record['validation_accuracy_pct']) > 1e-9:
        raise ValueError('Reloaded validation accuracy differs from the saved value.')
    return metrics, float((actual - expected).abs().max().item())


def save_checkpoint(path, model, metadata, dataset, validation_metrics):
    checkpoint = dict(metadata, schema_version=3, architecture=model.config,
                      state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                      parameter_count=parameter_count(model), weight_statistics=weight_statistics(model),
                      verification=verification_record(model, dataset, validation_metrics),
                      course_config=course_config(), environment=environment())
    with Path(path).open('xb') as stream:
        torch.save(checkpoint, stream)
    return checkpoint


def load_checkpoint(path, stage=None):
    from models import LeNet, validate_config
    path = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot read checkpoint {path.name}: {exc}') from exc
    required = {'schema_version', 'model_name', 'stage', 'architecture', 'state_dict',
                'course_config', 'split_sha256', 'data_manifest_sha256', 'parameter_count',
                'verification', 'origin', 'pruning', 'training_history', 'weight_statistics'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError('Missing Lab 3 checkpoint metadata. Import a Lab 2 file first.')
    if checkpoint['schema_version'] != 3 or checkpoint['stage'] not in ('source', 'pruned', 'deployment'):
        raise ValueError('Unsupported Lab 3 checkpoint version or stage.')
    if stage is not None and checkpoint['stage'] != stage:
        raise ValueError(f'This command needs a {stage} checkpoint, not {checkpoint["stage"]}.')
    if checkpoint['course_config'] != course_config():
        raise ValueError('Checkpoint settings differ from this Lab 3 project.')
    config = validate_config(checkpoint['architecture'], checkpoint['model_name'])
    model = LeNet(config).cpu()
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except RuntimeError as exc:
        raise ValueError(f'Checkpoint weights do not match its architecture/stage: {exc}') from exc
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite weights.')
    if parameter_count(model) != checkpoint['parameter_count']:
        raise ValueError('Checkpoint parameter count does not match its architecture.')
    model.eval()
    with torch.inference_mode():
        if tuple(model(torch.zeros(1, 1, 28, 28)).shape) != (1, 10):
            raise ValueError('Model must return ten logits.')
    if weight_statistics(model) != checkpoint['weight_statistics']:
        raise ValueError('Checkpoint weight statistics do not match its tensors.')
    if checkpoint['stage'] == 'deployment' and len(checkpoint['training_history']) != course_config()['epochs']:
        raise ValueError('Deployment checkpoint lacks the complete fine-tuning history.')
    return model, checkpoint, path


def checkpoint_provenance(checkpoint, path):
    return {key: checkpoint[key] for key in ('model_name', 'stage', 'architecture',
            'parameter_count', 'split_sha256', 'data_manifest_sha256', 'course_config',
            'origin', 'pruning')} | {'checkpoint_sha256': sha256(path)}


def run_cli(main):
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, NotImplementedError, RuntimeError,
            OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
