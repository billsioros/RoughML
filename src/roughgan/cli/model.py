import functools
import os
from pathlib import Path

import click
import plotly.express as px
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from roughgan.content.loss import ArrayGraph2DContentLoss
from roughgan.data.loaders import load_multiple_datasets_from_pt
from roughgan.data.transforms import To, View
from roughgan.models.cnn import CNNDiscriminator, CNNGenerator
from roughgan.training.epoch import per_epoch
from roughgan.training.flow import TrainingFlow


@click.group()
def cli():
    """An ML approach to generating logos from text."""


@cli.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True),
    help="Where to load the dataset from.",
)
@click.option(
    "-s",
    "--state",
    type=click.Path(exists=False),
    help="Where to store the model's state to.",
)
def train(dataset, state):
    """Train the model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_generator():
        return CNNGenerator.from_device(device)

    def get_discriminator(generator):
        return CNNDiscriminator.from_generator(generator)

    criterion = BCELoss().to(device)

    def logging_callback(config, logging_dir):
        level = config.handlers.file.level.lower()

        config.handlers.file.filename = logging_dir / f"{level}.log"

        return config

    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "CRITICAL").upper()

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"},
        },
        "handlers": {
            "default": {
                "level": LOGGING_LEVEL,
                "formatter": "standard",
                "class": "logging.StreamHandler",
            },
            "file": {
                "level": LOGGING_LEVEL,
                "formatter": "standard",
                "class": "logging.FileHandler",
            },
        },
        "loggers": {"": {"handlers": ["default", "file"], "level": LOGGING_LEVEL}},
    }

    training_flow = TrainingFlow(
        output_dir=Path.cwd() / "Output",
        logging={"config": LOGGING_CONFIG, "callback": logging_callback},
        training={
            "manager": {
                "benchmark": True,
                # Uncomment if you want to enable checkpointing
                "train_epoch": per_epoch,
                "log_every_n": 10,
                "criterion": {"instance": criterion},
                "n_epochs": 10,
                "train_ratio": 0.8,
                "optimizer": {
                    "type": Adam,
                    "params": {"lr": 0.0002, "betas": (0.5, 0.999)},
                },
                "dataloader": {
                    "batch_size": 256,
                    "shuffle": True,
                    "num_workers": 0,
                },
            },
        },
        content_loss={
            "type": ArrayGraph2DContentLoss,
            # Uncomment if you want to enable checkpointing
        },
        data={
            "loader": functools.partial(
                load_multiple_datasets_from_pt,
                Path.cwd() / "data",
                transforms=[To(device), View(1, 128, 128)],
                limit=(2, 10),
            ),
        },
        animation={
            "indices": [
                0,
            ],
            "save_path": "cnn_per_epoch_animation.mp4",
        },
        plot={
            "grayscale": {"limit": 10, "save_path_fmt": "grayscale/%s_%02d.png"},
            "surface": {"limit": 10, "save_path_fmt": "surface/%s_%02d.png"},
            "against": {"save_path_fmt": "against_%s.png"},
        },
        suppress_exceptions=False,
    )

    training_flow(get_generator, get_discriminator)


@cli.command()
@click.option(
    "-l",
    "--load-state",
    type=click.Path(exists=True),
    required=True,
    help="Where to load the model's state from.",
)
def generate(load_state):
    """Generate an image from a given text."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNNGenerator.from_state(load_state, device=device)

    model.eval()
    with torch.no_grad():
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=56,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = bert(input_ids=input_ids, attention_mask=attention_mask)

        image = model(outputs.last_hidden_state)

        def as_grayscale_image(array, save_path=None):
            fig = px.imshow(array)
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            if save_path is None:
                fig.show()
            else:
                with save_path.open("wb") as file:
                    fig.write_image(file)

        as_grayscale_image(
            image.cpu().detach().numpy().squeeze().reshape(256, 256, 3),
            save_path=Path.cwd() / f"{text}.png",
        )


if __name__ == "__main__":
    cli()
