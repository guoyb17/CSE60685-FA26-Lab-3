"""Branch construction and checks; complete weight transfer is in pruning_ops.py."""
import copy

import torch

from common import parameter_count
from models import BASELINE, LeNet, MODEL_CONFIGS
from pruning_ops import copy_compact_weights, select_conv2_channels


def valid_keep(keep, count):
    if not isinstance(keep, torch.Tensor) or keep.dtype != torch.long or keep.device.type != 'cpu':
        raise ValueError('keep must be a CPU int64 tensor.')
    if keep.ndim != 1 or keep.numel() != count or torch.unique(keep).numel() != count:
        raise ValueError('keep must contain the requested number of distinct indices.')
    if not ((keep >= 0) & (keep < 16)).all().item() or not torch.equal(keep, keep.sort().values):
        raise ValueError('Keep indices must be sorted and in 0..15.')


def check_compact(source, compact, keep, probes):
    """Compare with the full network whose removed pooled feature channels are zero."""
    source.eval()
    compact.eval()
    with torch.inference_mode():
        features = source.features(probes)
        mask = torch.zeros(16, dtype=features.dtype)
        mask[keep] = 1
        expected = source.classifier(torch.flatten(features * mask[None, :, None, None], 1))
        actual = compact(probes)
    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise ValueError('Compact logits differ from the matching masked-feature reference. Check all copied weights, biases and FC1 columns.')
    return float((actual - expected).abs().max().item())


def build_branch(source, name, probes):
    if source.config != BASELINE:
        raise ValueError('Every branch must start from the unchanged Lab 2 A architecture.')
    if name == 'control':
        return copy.deepcopy(source), {'method': 'none', 'scope': 'unpruned control'}
    if name not in ('s25', 's50'):
        raise ValueError('Unknown branch.')
    count = MODEL_CONFIGS[name]['c2']
    keep = select_conv2_channels(source.features[3].weight.detach(), count)
    valid_keep(keep, count)
    model = LeNet(MODEL_CONFIGS[name])
    copy_compact_weights(source, model, keep)
    if parameter_count(model) != {'s25': 36142, 's50': 27858}[name]:
        raise ValueError('Unexpected compact parameter count.')
    error = check_compact(source, model, keep, probes)
    return model, {'method': 'conv2_l1_channel', 'scope': 'Conv2 output filters and matching FC1 input columns',
                   'removed_conv2_channels': 16 - count, 'keep_indices': keep.tolist(),
                   'compact_max_abs_error': error}
