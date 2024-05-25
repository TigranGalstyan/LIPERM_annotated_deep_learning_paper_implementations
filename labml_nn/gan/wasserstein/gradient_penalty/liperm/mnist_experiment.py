"""
---
title: LIPERM WGAN-GP experiment with MNIST
summary: This experiment generates MNIST images using convolutional neural network.
---

# LIPERM WGAN-GP experiment with MNIST
"""
from typing import Any

import torch
import numpy as np
from labml_helpers.train_valid import BatchIndex
from scipy.stats import wasserstein_distance_nd
from torchvision.utils import make_grid
# from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from labml import tracker, monit, experiment

# Import configurations from [Wasserstein experiment](../experiment.html)
from labml_nn.gan.wasserstein.gradient_penalty.experiment import Configs as OriginalConfigs
from labml_helpers.optimizer import OptimizerConfigs
#

from labml.configs import option, calculate
from labml_nn.gan.wasserstein.gradient_penalty.liperm import InverseGeneratorLoss
from labml_nn.gan.wasserstein.gradient_penalty.liperm import MNISTInverseGenerator
from labml_nn.gan.wasserstein.gradient_penalty.liperm import get_pretrained_mnist_model


class Configs(OriginalConfigs):
    """
    ## Configuration class

    We extend [original GAN implementation](../../original/experiment.html) and override the discriminator (critic) loss
    calculation to include gradient penalty.
    """

    # Inverse penalty coefficient $\lambda$
    inverse_penalty_coefficient: float = 0.0
    #
    inverse_generator_loss = InverseGeneratorLoss()
    inverse_generator = 'cnn'
    inverse_generator_optimizer: torch.optim.Optimizer
    inception_metric = InceptionScore(normalize=True)

    pretrained_classification_model: torch.nn.Module

    def __init__(self):
        super(Configs, self).__init__()
        if self.device.type == 'cuda':
            # self.fid.cuda(self.device)
            self.inception_metric.cuda(self.device)

        self.pretrained_classification_model = get_pretrained_mnist_model(device=self.device)
        self.sample_train_images = torch.stack(self.collect_sample_images(train=True)).flatten(start_dim=1)
        self.sample_val_images = torch.stack(self.collect_sample_images(train=False)).flatten(start_dim=1)


    def collect_sample_images(self, train=True, num=256):
        samples = []
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.valid_dataset

        for i in np.random.permutation(np.arange(len(dataset)))[:num]:
            samples.append(self.train_dataset[i][0])
        return samples

    def calc_generator_loss(self, batch_size: int):
        """
        Calculate generator loss
        """
        latent = self.sample_z(batch_size)
        generated_images = self.generator(latent)
        logits = self.discriminator(generated_images)
        generated_latent = self.inverse_generator(generated_images)
        inverse_loss = self.inverse_generator_loss(latent, generated_latent)
        generator_loss = self.generator_loss(logits)

        loss = generator_loss + self.inverse_penalty_coefficient * inverse_loss

        # Log stuff
        tracker.add('generated', make_grid(generated_images[0:25], nrow=5, pad_value=1.0))
        tracker.add("loss.generator.", generator_loss)
        tracker.add("loss.inverse_generator.", inverse_loss)
        tracker.add("loss.overall.", loss)
        tracker.add("MC_score_max.", self.calc_mc_score(generated_images))
        tracker.add("InceptionScore.", self.calc_inception_score(generated_images))

        train_wd, val_wd = self.calculate_wd(generated_images)
        tracker.add("WassersteinDistanceTrain.", train_wd)
        tracker.add("WassersteinDistanceVal.", val_wd)

        return loss

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        Take a training step
        """

        # Set model states
        self.generator.train(self.mode.is_train)
        self.discriminator.train(self.mode.is_train)

        # Memory leak workaround
        if len(self.inception_metric.features) > 500:
            self.inception_metric.reset()

        # Get MNIST images
        data = batch[0].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Train the discriminator
        with monit.section("discriminator"):
            # Get discriminator loss
            loss = self.calc_discriminator_loss(data)

            # Train
            if self.mode.is_train:
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                if batch_idx.is_last:
                    tracker.add('discriminator', self.discriminator)
                self.discriminator_optimizer.step()

        # Train the generator once in every `discriminator_k`
        if batch_idx.is_interval(self.discriminator_k):
            with monit.section("generator"):
                loss = self.calc_generator_loss(data.shape[0])

                # Train
                if self.mode.is_train:
                    self.generator_optimizer.zero_grad()
                    self.inverse_generator_optimizer.zero_grad()
                    loss.backward()
                    if batch_idx.is_last:
                        tracker.add('generator', self.generator)
                    self.generator_optimizer.step()
                    self.inverse_generator_optimizer.step()

        tracker.save()

    def calc_mc_score(self, generated_images: torch.Tensor):

        classes = self.pretrained_classification_model(generated_images).argmax(dim=-1, keepdims=False)
        num_digits = 10
        batch_size = classes.size(0)
        c = classes.type(torch.IntTensor).flatten()
        bins = torch.bincount(c, minlength=num_digits)
        # deviation = ((bins - batch_size / 10) ** 2).mean().sqrt() / batch_size
        deviation = bins.max() / batch_size -  1 / 10
        return deviation

    def calc_inception_score(self, generated_images: torch.Tensor):

        samples = (generated_images + 1) / 2
        samples_3c = samples.tile(dims=(1, 3, 1, 1))

        self.inception_metric.update(samples_3c)
        inception_score = self.inception_metric.compute()[0]

        return inception_score

    def calculate_wd(self, generated):
        train_wd = wasserstein_distance_nd(generated.flatten(start_dim=1), self.sample_train_images)
        val_wd = wasserstein_distance_nd(generated.flatten(start_dim=1), self.sample_val_images)
        return train_wd, val_wd


@option(Configs.inverse_generator_optimizer)
def _inverse_generator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.inverse_generator.parameters()
    opt_conf.learning_rate = 2.5e-4
    # Setting exponent decay rate for first moment of gradient,
    # $\beta_1$ to `0.5` is important.
    # Default of `0.9` fails.
    opt_conf.betas = (0.5, 0.999)
    return opt_conf

calculate(Configs.inverse_generator, 'cnn', lambda c: MNISTInverseGenerator().to(c.device))


def main():
    # Create configs object
    conf = Configs()
    # Create experiment
    exp_name = 'MNIST_WGANGP_liperm0'
    experiment.create(name=exp_name, writers={'tensorboard'})
    # Override configurations
    experiment.configs(conf,
                       {
                           'discriminator': 'cnn',
                           'generator': 'cnn',
                           'label_smoothing': 0.01,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                           'discriminator_k': 5,
                           'train_batch_size': 256,
                           'valid_batch_size': 256,
                           'epochs': 25,
                           'inverse_penalty_coefficient': 0
                       })

    experiment.add_pytorch_models({
        'generator': conf.generator,
        'discriminator': conf.discriminator,
        'inverse_generator': conf.inverse_generator
    })
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
