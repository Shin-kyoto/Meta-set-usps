import numpy as np
import torch.utils.data as data
from PIL import Image
import os

class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if len(self.data.shape) > 3:
            img, target = self.data[:, :, :, index], int(self.targets[index])
        else:
            img, target = self.data[:, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = Image.fromarray(np.uint8(img)).convert('RGB')  # unit8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if len(self.data.shape) > 3:
            return self.data.shape[3]
        else:
            return self.data.shape[2]


class MNIST_bg(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, sample_file, label_file, transform=None, target_transform=None):
        super(MNIST_bg, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = np.load(str(root) + '//' + sample_file)
        self.targets = np.load(str(root) + '//' + label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[:, :, :, index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = Image.fromarray(np.uint8(img)).convert('L')  # unit8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[3]

class USPS(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        filename = 'usps.bz2' if train else 'usps.t.bz2'
        full_path = os.path.join(self.root, filename)

        import bz2
        with bz2.open(full_path) as fp:
            raw_data = [l.decode().split() for l in fp.readlines()]
            imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
            imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
            imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
            targets = [int(d[0]) - 1 for d in raw_data]

        self.data = imgs
        self.targets = targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)