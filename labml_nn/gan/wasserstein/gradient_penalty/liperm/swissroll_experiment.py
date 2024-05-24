"""
---
title: LIPERM WGAN-GP experiment with SwissRoll
summary: This experiment generates SwissRoll points.
---

# LIPERM WGAN-GP experiment with SwissRoll data
"""
from typing import Any

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from scipy.spatial import KDTree


from labml import tracker, monit, experiment
from labml.configs import option, calculate

from labml_nn.gan.wasserstein.gradient_penalty.liperm import InverseGeneratorLoss, \
    SwissRollGenerator, SwissRollDiscriminator, SwissRollInverseGenerator

from labml_nn.gan.wasserstein.gradient_penalty import GradientPenalty

from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.gan.wasserstein import GeneratorLoss, DiscriminatorLoss
from labml_nn.gan.wasserstein.gradient_penalty.liperm import SwissRollConfigs


class Configs(SwissRollConfigs, TrainValidConfigs):
    """
    ## Configurations

    This extends SwissRoll configurations to get the data loaders and Training and validation loop
    configurations to simplify our implementation.
    """

    device: torch.device = DeviceConfigs()
    epochs: int = 10
    latent_dim: int = 2

    is_save_models = False
    save_models_interval = 500
    discriminator: Module = 'mlp'
    generator: Module = 'mlp'
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    inverse_generator_optimizer: torch.optim.Adam
    generator_loss: GeneratorLoss = 'original'
    discriminator_loss: DiscriminatorLoss = 'original'
    label_smoothing: float = 0.0
    discriminator_k: int = 5

    # Gradient penalty coefficient $\lambda$
    gradient_penalty_coefficient: float = 1.0
    #
    gradient_penalty = GradientPenalty()

    # Inverse penalty coefficient $\lambda$
    inverse_penalty_coefficient: float = 1.0
    #
    inverse_generator_loss = InverseGeneratorLoss()
    inverse_generator = 'mlp'

    # For calculating minimum distance metric
    kdtree = None


    def init(self):
        """
        Initializations
        """
        self.state_modules = []
        self.kdtree = KDTree(self.train_dataset.data)

        hook_model_outputs(self.mode, self.generator, 'generator')
        hook_model_outputs(self.mode, self.discriminator, 'discriminator')
        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)
        tracker.set_image("generated", True)

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
        data = batch.to(self.device)

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
        tracker.add("loss.discriminator.wasserstein.", - loss_true - loss_false)
        tracker.add("loss.discriminator.", loss)

        return loss

    def calc_generator_loss(self, batch_size: int):
        """
        Calculate generator loss
        """
        latent = self.sample_z(batch_size)
        generated_points = self.generator(latent)
        logits = self.discriminator(generated_points)
        generated_latent = self.inverse_generator(generated_points)
        inverse_loss = self.inverse_generator_loss(latent, generated_latent)
        generator_loss = self.generator_loss(logits)

        loss = generator_loss + self.inverse_penalty_coefficient * inverse_loss

        # Log stuff

        generated_points_cpu = generated_points.detach().cpu()

        fig, ax = plt.subplots()
        ## workaround
        ax.scatter(self.train_dataset.data[::10, 0], self.train_dataset.data[::10, 1])
        ###
        ax.scatter(generated_points_cpu[:, 0], generated_points_cpu[:, 1])
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()

        tracker.add("generated", transforms.PILToTensor()(img))
        tracker.add("loss.generator.", generator_loss)
        tracker.add("loss.inverse_generator.", inverse_loss)
        tracker.add("loss.overall.", loss)
        tracker.add("dissimilarity_score", self.calculate_dissimiliarity_score(generated_points_cpu, skip_step=1))

        return loss

    def calculate_dissimiliarity_score(self, generated, skip_step=10):
        # calculates the minimum distances, higher the better
        distances, _ = self.kdtree.query(generated[::skip_step])
        return distances.mean()


calculate(Configs.generator, 'mlp', lambda c: SwissRollGenerator().to(c.device))
calculate(Configs.discriminator, 'mlp', lambda c: SwissRollDiscriminator().to(c.device))
calculate(Configs.inverse_generator, 'mlp', lambda c: SwissRollInverseGenerator().to(c.device))

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
    exp_name = 'SwisRoll_2.0'
    experiment.create(name=exp_name, writers={'tensorboard'})
    # Override configurations
    experiment.configs(conf,
                       {
                           'label_smoothing': 0.0,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                           'discriminator_k': 5,
                           'train_batch_size': 200,
                           'valid_batch_size': 200,
                           'epochs': 5000,
                           'inverse_penalty_coefficient': 2.0,
                       })

    experiment.add_pytorch_models({'generator': conf.generator,
                                   'discriminator': conf.discriminator,
                                   'inverse_generator': conf.inverse_generator})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
