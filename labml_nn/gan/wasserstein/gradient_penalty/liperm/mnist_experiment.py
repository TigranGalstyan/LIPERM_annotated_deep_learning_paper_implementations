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
    inception_metric = InceptionScore(normalize=True, splits=1)

    sample_train_images: torch.Tensor
    sample_val_images: torch.Tensor
    valid_loader_shuffle = True

    def __init__(self):
        super(Configs, self).__init__()

        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)
        tracker.set_image("Generated Images", True)

        if self.device.type == 'cuda':
            # self.fid.cuda(self.device)
            self.inception_metric.cuda(self.device)

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

        tracker.add("loss.generator.", generator_loss)
        tracker.add("loss.inverse_generator.", inverse_loss)
        tracker.add("loss.overall.", loss)

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

        if not self.mode.is_train and batch_idx.is_last:

            train_wd, val_wd, inception_score, generated_images = self.get_scores(batch_size=data.shape[0])
            tracker.add("WassersteinDistanceTrain.", train_wd)
            tracker.add("WassersteinDistanceVal.", val_wd)
            tracker.add('Generated Images', make_grid(generated_images[0:25, [0]], nrow=5, pad_value=1.0))
            tracker.add("InceptionScore.", inception_score)
        tracker.save()

    def get_scores(self, batch_size: int):
        with torch.no_grad():
            latent = self.sample_z(batch_size)
            batch = self.generator(latent)
            train_wd = wasserstein_distance_nd(batch.detach().cpu().flatten(start_dim=1)[::8], self.sample_train_images)
            val_wd = wasserstein_distance_nd(batch.detach().cpu().flatten(start_dim=1)[::8], self.sample_val_images)

            generated_images = batch  # Denormalization
            generated_images_3c = generated_images.tile(dims=(1, 3, 1, 1))

            self.inception_metric.update((generated_images_3c + 1 ) / 2)
            inception_score = self.inception_metric.compute()[0]
            self.inception_metric.reset()

        return train_wd, val_wd, inception_score, generated_images

@option(Configs.inverse_generator_optimizer)
def _inverse_generator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.inverse_generator.parameters()
    opt_conf.learning_rate = 1e-4
    # Setting exponent decay rate for first moment of gradient,
    # $\beta_1$ to `0.5` is important.
    # Default of `0.9` fails.
    opt_conf.betas = (0.5, 0.999)
    return opt_conf

@option(Configs.discriminator_optimizer)
def _discriminator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.discriminator.parameters()
    opt_conf.learning_rate = 1e-4
    # Setting exponent decay rate for first moment of gradient,
    # $\beta_1$ to `0.5` is important.
    # Default of `0.9` fails.
    opt_conf.betas = (0.5, 0.999)
    return opt_conf


@option(Configs.generator_optimizer)
def _generator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.generator.parameters()
    opt_conf.learning_rate = 1e-4
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
    exp_name = 'MNIST_WGANGP_liperm8.0'
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
                           'train_batch_size': 512,
                           'valid_batch_size': 512,
                           'epochs': 500,
                           'inverse_penalty_coefficient': 8.0
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
