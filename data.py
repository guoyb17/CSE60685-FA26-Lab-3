"""Fashion-MNIST loaders using the published training and validation indices."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import ROOT, course_config, read_json


def split_spec():
    return read_json(ROOT / 'assets' / 'split.json')


def training_sets(data_dir):
    dataset = datasets.FashionMNIST(root=str(data_dir), train=True, download=False,
                                   transform=transforms.ToTensor())
    spec = split_spec()
    return dataset, Subset(dataset, spec['train_indices']), Subset(dataset, spec['validation_indices'])


def test_set(data_dir):
    dataset = datasets.FashionMNIST(root=str(data_dir), train=False, download=False,
                                   transform=transforms.ToTensor())
    return dataset


def loader(dataset, training=False):
    cfg = course_config()
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(dataset, batch_size=cfg['batch_size'] if training else cfg['evaluation_batch_size'],
                      shuffle=training, num_workers=0, generator=generator, drop_last=False)


def benchmark_input(data_dir):
    dataset, _, _ = training_sets(data_dir)
    index = split_spec()['benchmark_index']
    tensor, label = dataset[index]
    return tensor.unsqueeze(0).contiguous(), index, label
