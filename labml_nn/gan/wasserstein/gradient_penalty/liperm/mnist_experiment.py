"""
---
title: LIPERM WGAN-GP experiment with MNIST
summary: This experiment generates MNIST images using convolutional neural network.
---

# LIPERM WGAN-GP experiment with MNIST
"""
from typing import Any

import torch
from labml_helpers.train_valid import BatchIndex
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from labml import experiment, tracker

# Import configurations from [Wasserstein experiment](../experiment.html)
from labml_nn.gan.wasserstein.gradient_penalty.experiment import Configs as OriginalConfigs
#
from labml.configs import calculate
from labml_nn.gan.wasserstein.gradient_penalty.liperm import InverseGeneratorLoss
from labml_nn.gan.wasserstein.gradient_penalty.liperm import MNISTInverseGenerator


class Configs(OriginalConfigs):
    """
    ## Configuration class

    We extend [original GAN implementation](../../original/experiment.html) and override the discriminator (critic) loss
    calculation to include gradient penalty.
    """

    # Inverse penalty coefficient $\lambda$
    inverse_penalty_coefficient: float = 1.0
    #
    inverse_generator_loss = InverseGeneratorLoss()
    inverse_generator = 'cnn'
    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False, feature=2048)
    inception_metric = InceptionScore(normalize=True)

    def __init__(self):
        super(Configs, self).__init__()
        if self.device.type == 'cuda':
            self.fid.cuda(self.device)
            self.inception_metric.cuda(self.device)

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
        samples_3c = samples.tile(dims=(1, 3, 1, 1))
        self.fid.update(samples_3c, real=False)
        fid_score = self.fid.compute()
        self.fid.reset()

        self.inception_metric.update(samples_3c)
        inception_score = self.inception_metric.compute()[0]

        return fid_score, inception_score

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        Take a training step
        """
        super().step(batch, batch_idx)
        batch_size = batch[0].shape[0]
        if tracker.get_global_step() < batch_idx.total * batch_size:
            # Get MNIST images
            data = batch[0].to(self.device)
            data_3c = torch.tile((data + 1.0) / 2.0, dims=(1, 3, 1, 1))
            self.fid.update(data_3c, real=True)

        # Compute metrics once in every `discriminator_k`
        if batch_idx.is_interval(self.discriminator_k):
            fid_score, inception_score = self.get_fid_and_inception_score(batch_size=batch_size)
            tracker.add("FrechetInceptionDistance.", fid_score)
            tracker.add("InceptionScore.", inception_score)


calculate(Configs.inverse_generator, 'cnn', lambda c: MNISTInverseGenerator().to(c.device))


def main():
    # Create configs object
    conf = Configs()
    # Create experiment
    exp_name = 'MNIST_DEBug'
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
                           'inverse_penalty_coefficient': 0.5
                       })

    # Start the experiment and run training loop
    # with experiment.record(name='MNIST_DEBUG', token='http://localhost:7778/api/v1/track?', writers={'tensorboard'}):
    experiment.add_pytorch_models({'generator': conf.generator})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
