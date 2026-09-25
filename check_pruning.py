"""Check the supplied pruning functions with known ranks and independent feature masking."""
import torch
from common import parameter_count, run_cli, setup_cpu
from models import BASELINE, LeNet
from pruning import build_branch
from pruning_ops import select_conv2_channels


def main():
    setup_cpu()
    weights = torch.arange(16, dtype=torch.float32)[:, None, None, None].expand(16, 6, 5, 5)
    if not torch.equal(select_conv2_channels(weights, 8), torch.arange(8, 16)):
        raise ValueError('Keep the largest L1 output-filter scores.')
    tied = torch.ones(16, 6, 5, 5)
    if not torch.equal(select_conv2_channels(tied, 12), torch.arange(12)):
        raise ValueError('Resolve equal scores by the smaller original channel index.')
    source = LeNet(BASELINE)
    probes = torch.cat([torch.zeros(1, 1, 28, 28), torch.rand(4, 1, 28, 28)])
    for name in ('s25', 's50'):
        compact, result = build_branch(source, name, probes)
        print(f'PASS {name}: {compact.config["c2"]} Conv2 channels; '
              f'{compact.classifier[0].in_features} FC1 inputs; {parameter_count(compact):,} parameters; '
              f'max logit error {result["compact_max_abs_error"]:.3g}')


if __name__ == '__main__':
    run_cli(main)
