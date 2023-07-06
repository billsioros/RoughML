import functools
from datetime import datetime
from pathlib import Path

import click
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from roughgan.data.loaders import load_dataset_from_pt
from roughgan.data.transforms import To, View
from roughgan.models import (
    CNNDiscriminator,
    CNNGenerator,
    PerceptronDiscriminator,
    PerceptronGenerator,
)
from roughgan.training.epoch import per_epoch
from roughgan.training.manager import TrainingManager
from roughgan.tuning.flow import TuningFlow
from roughgan.tuning.trial import trial_factory


def get_generator_discriminator_fns(model_name, device):
    get_generator, get_discriminator = None, None

    if model_name == "perceptron":

        def get_generator():
            return CNNGenerator.from_device(device)

        def get_discriminator(generator):
            return CNNDiscriminator.from_device(generator.device)

    elif model_name == "cnn":

        def get_generator():
            return PerceptronGenerator.from_device(device)

        def get_discriminator(generator):
            return PerceptronDiscriminator.from_generator(generator)

    return get_generator, get_discriminator


@click.command()
@click.option(
    "-d",
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The path of an existing dataset",
)
@click.option(
    "-m",
    "--model",
    "model",
    required=True,
    default="cnn",
    show_default=True,
    type=click.Choice(["perceptron", "cnn"], case_sensitive=False),
    help="The model pair to fine tune",
)
@click.option(
    "-d",
    "--debug",
    "debug",
    required=False,
    is_flag=True,
    show_default=True,
    help="Run in debug mode",
)
def tune(dataset_path, model, debug):
    """Perform hyper parameter tuning on a generator/discriminator pair."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = BCELoss().to(device)

    get_generator, get_discriminator = get_generator_discriminator_fns(model, device)

    tuning_flow = TuningFlow(
        get_generator=get_generator,
        get_discriminator=get_discriminator,
        trial_factory=trial_factory(TrainingManager),
        dataloader=functools.partial(
            load_dataset_from_pt,
            dataset_path,
            transforms=[To(device), View(1, 128, 128)],
            limit=10 if debug else None,
        ),
        parameters={
            "train_epoch": per_epoch,
            "criterion": {"instance": criterion},
            "optimizer": {"type": Adam},
        },
    )

    tuning_flow(get_generator, get_discriminator, num_samples=1 if debug else 100)

    csv_path = Path.cwd() / f"tune_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.csv"
    with csv_path.open("w") as file:
        tuning_flow.dataframe.to_csv(file)


if __name__ == "__main__":
    tune()
