"""Check CPU forward/backward execution before running the experiment."""
import platform
import torch
import torchvision
from common import run_cli, setup_cpu


def main():
    setup_cpu()
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model[0].weight.detach().clone()
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(torch.ones(3, 4)), torch.tensor([0, 1, 0]))
    loss.backward()
    optimizer.step()
    if not torch.isfinite(loss).item() or torch.equal(before, model[0].weight):
        raise ValueError('CPU forward/backward check failed.')
    print(f'Python {platform.python_version()}; PyTorch {torch.__version__}; torchvision {torchvision.__version__}')
    print('PASS: CPU forward/backward; one intra-op and one inter-op thread.')


if __name__ == '__main__':
    run_cli(main)
