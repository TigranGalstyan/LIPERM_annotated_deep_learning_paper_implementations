"""
---
title: LIPERM WGAN-GP experiment with CIFAR10
summary: This experiment generates CIFAR10 images using convolutional neural network.
---

# LIPERM WGAN-GP experiment with CIFAR10
"""
from typing import Any

import torch

from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from labml import tracker, monit, experiment
from labml.configs import option, calculate


from labml_nn.gan.wasserstein.gradient_penalty.liperm import InverseGeneratorLoss, \
    CIFAR10Generator, CIFAR10Discriminator, CIFAR10InverseGenerator

from labml_nn.gan.wasserstein.gradient_penalty import GradientPenalty


import torch.utils.data

from labml_helpers.datasets.cifar10 import CIFAR10Configs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
# Import [Wasserstein GAN losses](./index.html)
from labml_nn.gan.wasserstein import GeneratorLoss, DiscriminatorLoss


class Configs(CIFAR10Configs, TrainValidConfigs):
    """
    ## Configurations

    This extends CIFAR10 configurations to get the data loaders and Training and validation loop
    configurations to simplify our implementation.
    """

    device: torch.device = DeviceConfigs()
    dataset_transforms = 'cifar10_transforms'
    epochs: int = 10
    latent_dim: int = 128

    is_save_models = True
    discriminator: Module = 'conv'
    generator: Module = 'resnet'
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    inverse_generator_optimizer: torch.optim.Adam
    generator_loss: GeneratorLoss = 'original'
    discriminator_loss: DiscriminatorLoss = 'original'
    label_smoothing: float = 0.2
    discriminator_k: int = 1

    # Gradient penalty coefficient $\lambda$
    gradient_penalty_coefficient: float = 1.0
    #
    gradient_penalty = GradientPenalty()

    # Inverse penalty coefficient $\lambda$
    inverse_penalty_coefficient: float = 1.0
    #
    inverse_generator_loss = InverseGeneratorLoss()
    inverse_generator = 'cnn'
    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False, feature=2048)
    inception_metric = InceptionScore(normalize=True)

    def init(self):
        """
        Initializations
        """
        self.state_modules = []

        hook_model_outputs(self.mode, self.generator, 'generator')
        hook_model_outputs(self.mode, self.discriminator, 'discriminator')
        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)
        tracker.set_image("generated", True)

        if self.device.type == 'cuda':
            self.fid.cuda(self.device)
            self.inception_metric.cuda(self.device)

    def sample_z(self, batch_size: int):
        """
        $$z \sim p(z)$$
        """
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        Take a training step
        """

        # Set model states
        self.generator.train(self.mode.is_train)
        self.discriminator.train(self.mode.is_train)

        # Get MNIST images
        data = batch[0].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        if len(self.inception_metric.features) > 500:  # Memory leak workaround
            self.inception_metric.reset()

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

        batch_size = batch[0].shape[0]
        # if tracker.get_global_step() < batch_idx.total * batch_size:
        self.fid.update((data + 1.0) / 2.0, real=True)

        # Compute metrics once in every `discriminator_k`
        if batch_idx.is_interval(self.discriminator_k):
            fid_score, inception_score = self.get_fid_and_inception_score(batch_size=batch_size)
            tracker.add("FrechetInceptionDistance.", fid_score)
            tracker.add("InceptionScore.", inception_score)

        tracker.save()

    def calc_discriminator_loss(self, data: torch.Tensor):
        """
        This overrides the original discriminator loss calculation and
        includes gradient penalty.
        """
        # Require gradients on $x$ to calculate gradient penalty
        data.requires_grad_()
        # Sample $z \sim p(z)$
        latent = self.sample_z(data.shape[0])
        # $D(x)$
        f_real = self.discriminator(data)
        # $D(G_\theta(z))$
        f_fake = self.discriminator(self.generator(latent).detach())
        # Get discriminator losses
        loss_true, loss_false = self.discriminator_loss(f_real, f_fake)
        # Calculate gradient penalties in training mode
        if self.mode.is_train:
            gradient_penalty = self.gradient_penalty(data, f_real)
            tracker.add("loss.gp.", gradient_penalty)
            loss = loss_true + loss_false + self.gradient_penalty_coefficient * gradient_penalty
        # Skip gradient penalty otherwise
        else:
            loss = loss_true + loss_false

        # Log stuff
        tracker.add("loss.discriminator.true.", loss_true)
        tracker.add("loss.discriminator.false.", loss_false)
        tracker.add("loss.discriminator.", loss)

        return loss

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

        return loss

    def get_fid_and_inception_score(self, batch_size=100):
        with torch.no_grad():
            latent = self.sample_z(batch_size)
            generated_images = self.generator(latent)

        samples = (generated_images + 1) / 2
        self.fid.update(samples, real=False)
        fid_score = self.fid.compute()
        self.fid.reset()

        self.inception_metric.update(samples)
        inception_score = self.inception_metric.compute()[0]

        return fid_score, inception_score


calculate(Configs.generator, 'resnet', lambda c: CIFAR10Generator().to(c.device))
calculate(Configs.discriminator, 'conv', lambda c: CIFAR10Discriminator(c.latent_dim).to(c.device))
calculate(Configs.inverse_generator, 'cnn', lambda c: CIFAR10InverseGenerator(c.latent_dim).to(c.device))

# Set configurations options for Wasserstein GAN losses
calculate(Configs.generator_loss, 'wasserstein', lambda c: GeneratorLoss())
calculate(Configs.discriminator_loss, 'wasserstein', lambda c: DiscriminatorLoss())


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


def continue_experiment(run_uuid, exp_name='debug'):

    experiment.create(name=exp_name, writers={'tensorboard'})

    # Create configs
    conf = Configs()
    # Load custom configuration of the training run
    configs_dict = experiment.load_configs(run_uuid)
    # Set configurations
    configs_dict.update({'epochs': 300})
    experiment.configs(conf, configs_dict)

    # Initialize
    conf.init()

    # Set PyTorch modules for saving and loading
    experiment.add_pytorch_models({'generator': conf.generator,
                                   'discriminator': conf.discriminator,
                                   'inverse_generator': conf.inverse_generator})

    # Load training experiment
    experiment.load(run_uuid)

    with experiment.start():
        conf.run()


def main():
    # Create configs object
    conf = Configs()
    # Create experiment
    exp_name = 'CIFAR10_1.0'
    experiment.create(name=exp_name, writers={'tensorboard'})
    # Override configurations
    experiment.configs(conf,
                       {
                           'discriminator': 'conv',
                           'generator': 'resnet',
                           'label_smoothing': 0.0,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                           'discriminator_k': 5,
                           'train_batch_size': 256,
                           'valid_batch_size': 256,
                           'epochs': 300,
                           'inverse_penalty_coefficient': 8.0,
                       })

    experiment.add_pytorch_models({'generator': conf.generator,
                                   'discriminator': conf.discriminator,
                                   'inverse_generator': conf.inverse_generator})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
