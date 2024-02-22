r"""
---
title: Gradient Penalty for Wasserstein GAN (WGAN-GP)
summary: >
 An annotated PyTorch implementation/tutorial of
  Improved Training of Wasserstein GANs.
---

# Gradient Penalty for Wasserstein GAN (WGAN-GP)

This is an implementation of
[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

[WGAN](../index.html) suggests clipping weights to enforce Lipschitz constraint
on the discriminator network (critic).
This and other weight constraints like L2 norm clipping, weight normalization,
L1, L2 weight decay have problems:

1. Limiting the capacity of the discriminator
2. Exploding and vanishing gradients (without [Batch Normalization](../../../normalization/batch_norm/index.html)).

The paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
proposal a better way to improve Lipschitz constraint, a gradient penalty.

$$\mathcal{L}_{GP} = \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}}
\Big[ \big(\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 - 1\big)^2 \Big]
$$

where $\lambda$ is the penalty weight and

\begin{align}
x &\sim \mathbb{P}_r \\
z &\sim p(z) \\
\epsilon &\sim U[0,1] \\
\tilde{x} &\leftarrow G_\theta (z) \\
\hat{x} &\leftarrow \epsilon x + (1 - \epsilon) \tilde{x}
\end{align}

That is we try to keep the gradient norm $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2$ close to $1$.

In this implementation we set $\epsilon = 1$.

Here is the [code for an experiment](experiment.html) that uses gradient penalty.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from labml_helpers.module import Module
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms

from labml import lab
from labml.configs import BaseConfigs
from sklearn.datasets import make_swiss_roll


class InverseGeneratorLoss(Module):
    """
    ## Inverse Generator Loss

    """

    def forward(self, orig_latent: torch.Tensor, pred_latent: torch.Tensor):
        """
        """
        return torch.nn.functional.mse_loss(orig_latent, pred_latent)


class MNISTInverseGenerator(Module):
    """
    ### Convolutional Inverse Generator Network for MNIST
    """

    def __init__(self):
        super().__init__()
        # The input is $28 \times 28$ with one channel
        self.conv_layers = nn.Sequential(
            # This gives $14 \times 14$
            nn.Conv2d(1, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # This gives $7 \times 7$
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # This gives $3 \times 3$
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # # This gives $1 \times 1$
            nn.Conv2d(1024, 100, 3, 1, 0, bias=False),
        )
        self.linear = nn.Linear(100, 100)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class CIFAR10Discriminator(Module):
    def __init__(self, dim=128):
        super(CIFAR10Discriminator, self).__init__()
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*dim, 1)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4*4*4*self.dim)
        x = self.linear(x)
        return x


class ResidualBlock(Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class CIFAR10Generator(Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(CIFAR10Generator, self).__init__()

        self.preprocess = nn.Sequential(
            nn.Linear(128, 3 * 8 * 8),
            nn.ReLU(True),
        )

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        img = self.preprocess(x).view(-1, 3, 8, 8)
        out1 = self.conv1(img)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class CIFAR10InverseGenerator(Module):
    def __init__(self, dim=128):
        super(CIFAR10InverseGenerator, self).__init__()
        self.dim = dim
        preprocess = nn.Sequential(
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
            nn.Linear(4 * 4 * 4 * dim, 128),
        )

        block1 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(2 * dim),
            nn.Conv2d(2 * dim, 4 * dim, 2, stride=2),
        )
        block2 = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, 2 * dim, 2, stride=2),
        )
        deconv_out = nn.Conv2d(3, dim, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out

    def forward(self, x):
        x = self.deconv_out(x)
        x = self.block2(x)
        x = self.block1(x)
        x = x.view(-1, 4 * 4 * 4 * self.dim)
        x = self.preprocess(x)

        return x


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


#  Swiss Roll Experiment


class SwissRollDataset(Dataset):

    def __init__(self, size):
        self.data = data = make_swiss_roll(
                n_samples=size,
                noise=1.5,
                random_state=1234
            )[0].astype('float32')[:, (0, 2)]
        self.data = self.data / 7.5  # stdev plus a little
        self.len = size

    def __getitem__(self, index) -> torch.Tensor:
        return data[index]

    def __getitems__(self, indices: List) -> List[T_co]:
    Not implemented to prevent false-positives in fetcher check in
    torch.utils.data._utils.fetch._MapDatasetFetcher

    def __len__(self):
        return self.len


def _dataset(is_train, transform):
    return datasets.CIFAR10(str(lab.get_data_path()),
                            train=is_train,
                            download=True,
                            transform=transform)


class SwissRollConfigs(BaseConfigs):
    """
    Configurable SwissRoll data set.

    Arguments:
        dataset_name (str): name of the data set, ``CIFAR10``
        dataset_transforms (torchvision.transforms.Compose): image transformations
        train_dataset (torchvision.datasets.CIFAR10): training dataset
        valid_dataset (torchvision.datasets.CIFAR10): validation dataset

        train_loader (torch.utils.data.DataLoader): training data loader
        valid_loader (torch.utils.data.DataLoader): validation data loader

        train_batch_size (int): training batch size
        valid_batch_size (int): validation batch size

        train_loader_shuffle (bool): whether to shuffle training data
        valid_loader_shuffle (bool): whether to shuffle validation data
    """
    dataset_name: str = 'SwissRoll'
    train_dataset: datasets.CIFAR10
    valid_dataset: datasets.CIFAR10

    train_loader: DataLoader
    valid_loader: DataLoader

    train_batch_size: int = 64
    valid_batch_size: int = 1024

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False



@CIFAR10Configs.calc(CIFAR10Configs.train_dataset)
def cifar10_train_dataset(c: CIFAR10Configs):
    return _dataset(True, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.valid_dataset)
def cifar10_valid_dataset(c: CIFAR10Configs):
    return _dataset(False, c.dataset_transforms)


@CIFAR10Configs.calc(CIFAR10Configs.train_loader)
def cifar10_train_loader(c: CIFAR10Configs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@CIFAR10Configs.calc(CIFAR10Configs.valid_loader)
def cifar10_valid_loader(c: CIFAR10Configs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


CIFAR10Configs.aggregate(CIFAR10Configs.dataset_name, 'CIFAR10',
                       (CIFAR10Configs.dataset_transforms, 'cifar10_transforms'),
                       (CIFAR10Configs.train_dataset, 'cifar10_train_dataset'),
                       (CIFAR10Configs.valid_dataset, 'cifar10_valid_dataset'),
                       (CIFAR10Configs.train_loader, 'cifar10_train_loader'),
                       (CIFAR10Configs.valid_loader, 'cifar10_valid_loader'))
