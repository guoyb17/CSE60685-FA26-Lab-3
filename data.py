"""Verified Fashion-MNIST files and a published, disjoint stratified split."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import ROOT, course_config, read_json, sha256


def split_spec():
    return read_json(ROOT / 'assets' / 'split.json')


def verify_files(data_dir):
    raw = Path(data_dir) / 'FashionMNIST' / 'raw'
    for entry in read_json(ROOT / 'assets' / 'data_manifest.json')['files']:
        path = raw / entry['raw_name']
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run python prepare_data.py first.')
        if sha256(path) != entry['raw_sha256']:
            raise ValueError(f'Dataset checksum mismatch: {path}. Preserve it for diagnosis and prepare a new data directory.')


def validate_split(spec, targets):
    cfg = course_config()
    train, validation = spec['train_indices'], spec['validation_indices']
    if len(train) != cfg['train_size'] or len(validation) != cfg['validation_size']:
        raise ValueError('Split sizes differ from course_config.json.')
    for name, indices in [('training', train), ('validation', validation)]:
        if any(type(i) is not int or i < 0 or i >= len(targets) for i in indices):
            raise ValueError(f'Invalid {name} index.')
        if len(set(indices)) != len(indices):
            raise ValueError(f'Duplicate {name} indices.')
        counts = torch.bincount(targets[indices], minlength=10)
        if counts.tolist() != [len(indices) // 10] * 10:
            raise ValueError(f'The {name} split is not equally stratified across ten classes.')
    if set(train) & set(validation):
        raise ValueError('Training and validation indices overlap.')
    if spec['benchmark_index'] not in validation:
        raise ValueError('The benchmark image must belong to the validation split.')


def training_sets(data_dir, verify=True):
    if verify:
        verify_files(data_dir)
    dataset = datasets.FashionMNIST(root=str(data_dir), train=True, download=False,
                                   transform=transforms.ToTensor())
    if len(dataset) != 60000 or tuple(dataset.data.shape[1:]) != (28, 28):
        raise ValueError('Expected 60,000 official 28x28 Fashion-MNIST training images.')
    spec = split_spec()
    validate_split(spec, dataset.targets)
    return dataset, Subset(dataset, spec['train_indices']), Subset(dataset, spec['validation_indices'])


def test_set(data_dir):
    verify_files(data_dir)
    dataset = datasets.FashionMNIST(root=str(data_dir), train=False, download=False,
                                   transform=transforms.ToTensor())
    if len(dataset) != 10000:
        raise ValueError('The final evaluation must use all 10,000 official test images.')
    return dataset


def loader(dataset, training=False):
    cfg = course_config()
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(dataset, batch_size=cfg['batch_size'] if training else cfg['evaluation_batch_size'],
                      shuffle=training, num_workers=0, generator=generator, drop_last=False)


def check_checkpoint_data(checkpoint):
    for key, filename in [('split_sha256', 'split.json'), ('data_manifest_sha256', 'data_manifest.json')]:
        if checkpoint[key] != sha256(ROOT / 'assets' / filename):
            raise ValueError(f'Checkpoint {key} differs from the installed course assets.')


def benchmark_input(data_dir):
    dataset, _, _ = training_sets(data_dir)
    index = split_spec()['benchmark_index']
    tensor, label = dataset[index]
    return tensor.unsqueeze(0).contiguous(), index, label
