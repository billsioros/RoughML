import logging
from pathlib import Path

import click
import plotly.graph_objects as go
from rich.logging import RichHandler

from roughgan import metrics, plot
from roughgan.data.sets import NanoroughSurfaceDataset

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "-d",
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The path of an existing dataset",
)
@click.option(
    "-s",
    "--save",
    "save_path",
    required=False,
    type=click.Path(exists=False, dir_okay=False),
    help="Where to save the resulting figure",
)
@click.option(
    "-l",
    "--log",
    "logging_level",
    required=False,
    type=click.Choice(
        [
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARNING",
            "WARN",
            "INFO",
            "DEBUG",
            "NOTSET",
        ],
        case_sensitive=False,
    ),
    help="Specify the logging level",
    default="NOTSET",
    show_default=True,
)
@click.pass_context
def visualize(ctx, dataset_path, save_path, logging_level):
    """Various methods of visualizing nanorough surfaces."""
    ctx.obj = {"dataset_path": dataset_path, "save_path": save_path}

    logging.basicConfig(
        level=logging_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@visualize.command()
@click.pass_obj
def correlation(config):
    """Plot the correlation of all surfaces included in the provided dataset."""
    dataset_path, save_path = config["dataset_path"], config["save_path"]

    try:
        dataset = NanoroughSurfaceDataset.from_pt(dataset_path)

        data = []
        for index, surface in enumerate(dataset.surfaces):
            x, y = metrics.correlation(surface.numpy())

            data.append(go.Scatter(x=x, y=y, name="surface %04d" % (index + 1,)))

        fig = go.Figure(
            data=data,
            layout_title_text=f"The correlation per surface of dataset {dataset_path}",
        )

        if save_path is None:
            fig.show()
        else:
            with Path(save_path).open("wb") as file:
                fig.write_image(file)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception(e)


@visualize.command()
@click.option(
    "-i",
    "--index",
    required=True,
    type=click.INT,
    help="The index of the surface with regards to the dataset",
)
@click.pass_obj
def as_3d_surface(config, index):
    """Plot a surface of the provided dataset as a 3D surface."""
    dataset_path, save_path = config["dataset_path"], config["save_path"]

    try:
        dataset = NanoroughSurfaceDataset.from_pt(dataset_path)

        plot.as_3d_surface(dataset.surfaces[index], save_path=save_path)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception(e)


@visualize.command()
@click.option(
    "-i",
    "--index",
    required=True,
    type=click.INT,
    help="The index of the surface with regards to the dataset",
)
@click.pass_obj
def grayscale(config, index):
    """Plot a surface of the provided dataset as a grayscale image."""
    dataset_path, save_path = config["dataset_path"], config["save_path"]

    try:
        dataset = NanoroughSurfaceDataset.from_pt(dataset_path)

        plot.as_grayscale_image(dataset.surfaces[index], save_path=save_path)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    visualize()
