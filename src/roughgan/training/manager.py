import logging

import torch
from tqdm import tqdm

from roughgan.shared.configuration import Configuration
from roughgan.shared.decorators import benchmark
from roughgan.training.split import train_test_dataloaders

logger = logging.getLogger(__name__)


class TrainingManager(Configuration):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not hasattr(self, "benchmark"):
            self.benchmark = False

        if not hasattr(self, "log_every_n"):
            self.log_every_n = None

        if not hasattr(self, "checkpoint"):
            self.checkpoint = Configuration(directory=None, multiple=False)

        if not hasattr(self, "content_loss"):
            self.content_loss = None

        if self.content_loss is None:
            self.criterion.weight = 1

        if not hasattr(self.criterion, "weight"):
            self.criterion.weight = 0.5

        self.content_loss_weight = 1 - self.criterion.weight

        if not hasattr(self, "fixed_noise_dim"):
            self.fixed_noise_dim = 64

    def __call__(self, generator, discriminator, dataset):
        fixed_noise = torch.randn(
            self.fixed_noise_dim,
            *generator.feature_dims,
            dtype=generator.dtype,
            device=generator.device,
        )

        train_dataloader, _ = train_test_dataloaders(
            dataset,
            train_ratio=self.train_ratio,
            **self.dataloader.to_dict(),
        )

        optimizer_generator = self.optimizer.type(
            generator.parameters(),
            **self.optimizer.params.to_dict(),
        )

        optimizer_discriminator = self.optimizer.type(
            discriminator.parameters(),
            **self.optimizer.params.to_dict(),
        )

        train_epoch_f = self.train_epoch

        if self.benchmark is True and logger.level <= logging.INFO:
            train_epoch_f = benchmark(train_epoch_f)

        min_generator_loss = float("inf")
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                content_loss,
                vector_content_loss,
            ) = train_epoch_f(
                generator,
                discriminator,
                train_dataloader,
                optimizer_generator,
                optimizer_discriminator,
                self.criterion.instance,
                content_loss_fn=self.content_loss,
                vector_content_loss_fn=self.vector_content_loss,
                loss_weights=(self.content_loss_weight, self.criterion.weight),
                log_every_n=self.log_every_n,
            )

            if self.checkpoint.directory is not None and generator_loss < min_generator_loss:
                min_generator_loss = generator_loss

                generator_mt, discriminator_mt = (
                    f"{generator.__class__.__name__}",
                    f"{discriminator.__class__.__name__}",
                )

                if self.checkpoint.multiple is True:
                    generator_mt += f"_{epoch:03d}"
                    discriminator_mt += f"_{epoch:03d}"

                torch.save(
                    generator.state_dict(),
                    self.checkpoint.directory / f"{generator_mt}.pt",
                )

                torch.save(
                    discriminator.state_dict(),
                    self.checkpoint.directory / f"{discriminator_mt}.pt",
                )

            with torch.no_grad():
                fixed_fake = generator(fixed_noise).detach().cpu()

            logger.info(
                "Epoch: %02d, Generator Loss: %7.3f (Content Loss: %7.3f, Vector Loss: %7.3f), Discriminator Loss: %7.3f",
                epoch,
                generator_loss,
                content_loss,
                vector_content_loss,
                discriminator_loss,
            )

            logger.info(
                "Epoch: %02d, Discriminator Output: [Real: %7.3f, Fake: %7.3f]",
                epoch,
                discriminator_output_real,
                discriminator_output_fake,
            )

            yield (
                generator_loss,
                discriminator_loss,
                discriminator_output_real,
                discriminator_output_fake,
                content_loss,
                vector_content_loss,
                fixed_fake,
            )
