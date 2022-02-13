import logging
import logging.config
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import torch

from roughml.content.loss import VectorSpaceContentLoss
from roughml.plot import animate_epochs, as_3d_surface, as_grayscale_image, plot_against
from roughml.shared.configuration import Configuration
from roughml.shared.context_managers import ExceptionLoggingHandler
from roughml.training.manager import TrainingManager

logger = logging.getLogger(__name__)


class TrainingFlow(Configuration):
    def __init__(self, **props):
        super().__init__(**props)

        if not hasattr(self.training, "manager"):
            self.training.manager = Configuration()

        if not hasattr(self.training, "callbacks"):
            self.training.callbacks = []

        if not hasattr(self, "plot"):
            self.plot = Configuration(
                gray_scale={"limit": None, "save_path_fmt": None},
                surface={"limit": None, "save_path_fmt": None},
                against={"save_path_fmt": None},
            )

        if not hasattr(self, "animation"):
            self.animation = Configuration(
                indices=[
                    0,
                ],
                save_path=None,
            )

        if not hasattr(self.animation, "parameters"):
            self.animation.parameters = Configuration(
                interval=1000,
                repeat_delay=1000,
                blit=True,
                fps=15,
                bitrate=1800,
            )

        if not hasattr(self, "suppress_exceptions"):
            self.suppress_exceptions = True

        if not hasattr(self, "content_loss"):
            self.content_loss = Configuration(type=None, cache=None)

    def __call__(self, get_generator, get_discriminator):
        for path, dataset in self.data.loader():
            logger.info("Instantiating generator and discriminator")

            generator = get_generator()
            discriminator = get_discriminator(generator)

            start_time = time()

            with ExceptionLoggingHandler(
                logger, suppress_exceptions=self.suppress_exceptions
            ) as exception_logging_handler:
                dataset_output_dir = self.process_dataset(
                    generator, discriminator, path, dataset
                )

            end_time = time()

            for callback in filter(None, self.training.callbacks):
                callback(
                    log_file=self.logging.config.handlers.file.filename,
                    dataset=path,
                    generator=generator.__class__.__name__,
                    discriminator=discriminator.__class__.__name__,
                    elapsed_time=str(timedelta(seconds=end_time - start_time)),
                    succeeded=exception_logging_handler.success,
                )

            checkpoint_dir = dataset_output_dir / "Checkpoint"

            generator_mt, discriminator_mt = (
                f"{generator.__class__.__name__}",
                f"{discriminator.__class__.__name__}",
            )

            torch.save(
                generator.state_dict(),
                checkpoint_dir / f"{generator_mt}.pt",
            )

            torch.save(
                discriminator.state_dict(),
                checkpoint_dir / f"{discriminator_mt}.pt",
            )

    def process_dataset(self, generator, discriminator, path, dataset):
        dataset_output_dir = (
            self.output_dir
            / path.stem
            / f"{generator.__class__.__name__}_{discriminator.__class__.__name__}"
            / datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        )

        checkpoint_dir = dataset_output_dir / "Checkpoint"
        checkpoint_dir.mkdir(parents=True)

        if hasattr(self.training.manager, "checkpoint"):
            self.training.manager.checkpoint.directory = checkpoint_dir

        content_loss_cache = None
        if getattr(self.content_loss, "cache", None) is not None:
            content_loss_cache = checkpoint_dir / self.content_loss.cache

        logging_dir = dataset_output_dir / "Logging"
        logging_dir.mkdir(parents=True)

        self.logging.config = self.logging.callback(self.logging.config, logging_dir)

        logging.config.dictConfig(self.logging.config.to_dict())

        plotting_dir = dataset_output_dir / "Plots"
        plotting_dir.mkdir(parents=True)

        animation_save_path = plotting_dir / self.animation.save_path
        self.plot.save_directory = plotting_dir

        logger.info(
            "Running the flow on a %s/%s pair over dataset %s",
            generator.__class__.__name__,
            discriminator.__class__.__name__,
            path,
        )

        vector_content_loss = VectorSpaceContentLoss(surfaces=dataset.surfaces)

        if hasattr(self.training.manager, "content_loss"):
            training_manager = TrainingManager(
                vector_content_loss=vector_content_loss,
                **self.training.manager.to_dict(),
            )
        else:
            content_loss = None
            if content_loss_cache is not None:
                if content_loss_cache.is_file():
                    content_loss = self.content_loss.type.from_pickle(
                        content_loss_cache
                    )
                else:
                    content_loss = self.content_loss.type(surfaces=dataset.surfaces)
                    content_loss.to_pickle(content_loss_cache)
            else:
                if getattr(self.content_loss, "type", None) is not None:
                    content_loss = self.content_loss.type(surfaces=dataset.surfaces)

            training_manager = TrainingManager(
                vector_content_loss=vector_content_loss,
                content_loss=content_loss,
                **self.training.manager.to_dict(),
            )

        (
            generator_losses,
            discriminator_losses,
            discriminator_output_reals,
            discriminator_output_fakes,
            content_losses,
            vector_content_losses,
            fixed_fakes,
        ) = list(zip(*list(training_manager(generator, discriminator, dataset))))

        (
            save_path_gen_vs_dis_loss,
            save_path_dis_output,
            save_path_bce_vs_con_loss,
            save_path_vector_loss,
        ) = (None, None, None, None)
        if self.plot.against.save_path_fmt is not None:
            save_path_gen_vs_dis_loss = self.plot.against.save_path_fmt % (
                "gen_vs_dis_loss",
            )
            save_path_bce_vs_con_loss = self.plot.against.save_path_fmt % (
                "bce_vs_con_loss",
            )
            save_path_dis_output = self.plot.against.save_path_fmt % (
                "discriminator_output",
            )
            save_path_vector_loss = self.plot.against.save_path_fmt % (
                "content_vs_vector_loss",
            )

        pd.DataFrame(
            data=np.array(
                [
                    generator_losses,
                    discriminator_losses,
                    discriminator_output_reals,
                    discriminator_output_fakes,
                    content_losses,
                    vector_content_losses,
                ]
            ).T,
            columns=[
                "Generator Loss",
                "Discriminator Loss",
                "Discriminator Output (Real)",
                "Discriminator Output (Fake)",
                f"Content Loss ({self.content_loss.type.__name__ if hasattr(self.content_loss, 'type') else 'None'})",
                "Content Loss (VectorSpaceContentLoss)",
            ],
        ).to_csv(str(checkpoint_dir / "per_epoch_data.csv"))

        plot_against(
            generator_losses,
            discriminator_losses,
            title="Mean Generator vs Discriminator loss per epoch",
            xlabel="Epoch",
            ylabel="Loss",
            labels=("Generator", "Discriminator"),
            save_path=self.plot.save_directory / save_path_gen_vs_dis_loss,
        )

        plot_against(
            discriminator_output_reals,
            discriminator_output_fakes,
            title="Mean Discriminator Output per epoch",
            xlabel="Epoch",
            ylabel="Discriminator Output",
            labels=("Real Data", "Generator Data"),
            save_path=self.plot.save_directory / save_path_dis_output,
        )

        plot_against(
            generator_losses,
            content_losses,
            title="Mean Generator Total loss vs Content loss per epoch",
            xlabel="Epoch",
            ylabel="Loss",
            labels=("BCE + Content Loss", "Content Loss"),
            save_path=self.plot.save_directory / save_path_bce_vs_con_loss,
        )

        plot_against(
            vector_content_losses,
            content_losses,
            title="Vector vs Content loss per epoch",
            xlabel="Epoch",
            ylabel="Loss",
            labels=("Vector Loss", "Content Loss"),
            save_path=self.plot.save_directory / save_path_vector_loss,
        )

        animate_epochs(
            fixed_fakes,
            indices=self.animation.indices,
            save_path=animation_save_path,
            **self.animation.parameters.to_dict(),
        )

        if self.plot.save_directory is not None:
            (self.plot.save_directory / self.plot.grayscale.save_path_fmt).parent.mkdir(
                parents=True, exist_ok=True
            )

            if self.plot.grayscale.save_path_fmt is not None:
                for i in range(self.plot.grayscale.limit):
                    true_surface = dataset.surfaces[i].squeeze()
                    fake_surface = fixed_fakes[-1][i].squeeze()

                    as_grayscale_image(
                        true_surface,
                        self.plot.save_directory
                        / (self.plot.grayscale.save_path_fmt % ("true", i)),
                    )
                    as_grayscale_image(
                        fake_surface,
                        self.plot.save_directory
                        / (self.plot.grayscale.save_path_fmt % ("fake", i)),
                    )

            if self.plot.surface.save_path_fmt is not None:
                (
                    self.plot.save_directory / self.plot.surface.save_path_fmt
                ).parent.mkdir(parents=True, exist_ok=True)

                for i in range(self.plot.surface.limit):
                    true_surface = dataset.surfaces[i].squeeze()
                    fake_surface = fixed_fakes[-1][i].squeeze()

                    as_3d_surface(
                        true_surface,
                        self.plot.save_directory
                        / (self.plot.surface.save_path_fmt % ("true", i)),
                    )
                    as_3d_surface(
                        fake_surface,
                        self.plot.save_directory
                        / (self.plot.surface.save_path_fmt % ("fake", i)),
                    )

        return dataset_output_dir
