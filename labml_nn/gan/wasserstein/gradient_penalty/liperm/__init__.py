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
from collections import OrderedDict
from typing import List


import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.utils.data import DataLoader, Dataset

from labml_helpers.module import Module

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

    transform = torch.from_numpy

    def __init__(self, size, train: bool = True):
        self.data = make_swiss_roll(
                n_samples=size,
                noise=1.5,
                random_state=1234 * train
            )[0].astype('float32')[:, (0, 2)]
        self.data = self.data / 7.5  # stdev plus a little
        self.len = size

    def __getitem__(self, index) -> torch.Tensor:
        return self.transform(self.data[index])

    # def __getitems__(self, indices: List) -> torch.Tensor:
    #     return self.transform(self.data[indices])

    def __len__(self):
        return self.len


def _swissroll_dataset(size, is_train):
    return SwissRollDataset(size, is_train)


class SwissRollConfigs(BaseConfigs):
    """
    Configurable SwissRoll data set.
    """
    dataset_name: str = 'SwissRoll'
    train_dataset: SwissRollDataset
    valid_dataset: SwissRollDataset

    train_loader: DataLoader
    valid_loader: DataLoader

    train_data_size: int = 2000
    val_data_size: int = 2000

    train_batch_size: int = 200
    valid_batch_size: int = 200

    train_loader_shuffle: bool = True
    valid_loader_shuffle: bool = False


@SwissRollConfigs.calc(SwissRollConfigs.train_dataset)
def swissroll_train_dataset(c: SwissRollConfigs):
    return _swissroll_dataset(c.train_data_size, True)


@SwissRollConfigs.calc(SwissRollConfigs.valid_dataset)
def swissroll_valid_dataset(c: SwissRollConfigs):
    return _swissroll_dataset(c.val_data_size, False)


@SwissRollConfigs.calc(SwissRollConfigs.train_loader)
def swissroll_train_loader(c: SwissRollConfigs):
    return DataLoader(c.train_dataset,
                      batch_size=c.train_batch_size,
                      shuffle=c.train_loader_shuffle)


@SwissRollConfigs.calc(SwissRollConfigs.valid_loader)
def swissroll_valid_loader(c: SwissRollConfigs):
    return DataLoader(c.valid_dataset,
                      batch_size=c.valid_batch_size,
                      shuffle=c.valid_loader_shuffle)


SwissRollConfigs.aggregate(SwissRollConfigs.dataset_name, 'SwissRoll',
                           (SwissRollConfigs.train_dataset, 'swissroll_train_dataset'),
                           (SwissRollConfigs.valid_dataset, 'swissroll_valid_dataset'),
                           (SwissRollConfigs.train_loader, 'swissroll_train_loader'),
                           (SwissRollConfigs.valid_loader, 'swissroll_valid_loader'))


class SwissRollGenerator(Module):

    def __init__(self):
        super(SwissRollGenerator, self).__init__()
        main = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 2),
        )

        self.main = main

    def forward(self, x):
        out = self.main(x)
        return out


class SwissRollInverseGenerator(Module):
    def __init__(self):
        super(SwissRollInverseGenerator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2),
        )
        # self.apply(_weights_init)
        self.main = main

    def forward(self, x):
        out = self.main(x)
        return out


class SwissRollDiscriminator(nn.Module):

    def __init__(self):
        super(SwissRollDiscriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

class PretrainedMnist(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(PretrainedMnist, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.input_dims
        return self.model.forward(x)


def get_pretrained_mnist_model(device, input_dims=784, n_hiddens=(256, 256), n_class=10):
    model = PretrainedMnist(input_dims, n_hiddens, n_class)
    m = model_zoo.load_url(model_urls['mnist'])
    state_dict = m.state_dict() if isinstance(m, nn.Module) else m
    model.load_state_dict(state_dict)
    return model.to(device)
