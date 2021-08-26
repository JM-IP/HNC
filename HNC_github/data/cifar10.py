from importlib import import_module

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def get_loader(args, kwargs):
    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    #norm_mean = [0.49139968, 0.48215827, 0.44653124]
    #norm_std = [0.24703233, 0.24348505, 0.26158768]
    loader_train = None

    if not args.test_only:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.CIFAR10(
                root=args.dir_data,
                train=True,
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.CIFAR10(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=500, shuffle=False, **kwargs
    )

    return loader_train, loader_test

