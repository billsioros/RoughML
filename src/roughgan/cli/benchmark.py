import logging
import time

import click
import numpy as np
import pandas as pd
from rich.logging import RichHandler

from roughgan.content.loss import (
    ArrayGraph2DContentLoss,
    HPG2DContentLoss,
    NGramGraphContentLoss,
    VectorSpaceContentLoss,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

name_to_similarities = {
    "A2G": ArrayGraph2DContentLoss,
    "NGG": NGramGraphContentLoss,
    "HPG": HPG2DContentLoss,
    "FHS": VectorSpaceContentLoss,
}

similarity_to_names = {
    ArrayGraph2DContentLoss: "A2G",
    NGramGraphContentLoss: "NGG",
    HPG2DContentLoss: "HPG",
    VectorSpaceContentLoss: "FHS",
}


@click.command()
@click.option(
    "-s",
    "--similarity",
    required=False,
    type=click.Choice(name_to_similarities.keys(), case_sensitive=False),
    help="The content similarity metric that you would like to benchmark",
)
def benchmark(similarity):
    similarity_clses = name_to_similarities.values()

    if similarity is not None:
        similarity_name = similarity.upper()

        similarity_clses = [
            name_to_similarities[similarity_name],
        ]

    df = pd.DataFrame(
        columns=[
            "m_similarity",
            "n_surfaces",
            "s_dim",
            "e_training_time",
            "e_evaluation_time",
            "e_training_time_ratio",
            "e_evaluation_time_ratio",
        ],
    )

    for similarity_cls in similarity_clses:
        similarity_name = similarity_to_names[similarity_cls]
        for number_of_surfaces in [2 * i for i in range(1, 6)]:
            previously_elapsed_training_time, previously_elapsed_evaluation_time = 1, 1
            for dim_size in [2**i for i in range(1, 6)]:
                logger.info(
                    "Training and evaluating {0} for {1:02d} surfaces of size {2:03d}x{2:03d}".format(
                        similarity_name,
                        number_of_surfaces,
                        dim_size,
                    ),
                )

                surfaces = np.random.random((number_of_surfaces, dim_size, dim_size)) * 255

                start_time = time.time()

                content_similarity = similarity_cls(surfaces=surfaces)

                elapsed_training_time = time.time() - start_time

                logger.info(
                    "{} training took {:07.3f}s ({:09.3f} slower than dim_size={:02d})".format(
                        similarity_name,
                        elapsed_training_time,
                        elapsed_training_time / previously_elapsed_training_time,
                        dim_size // 2,
                    ),
                )

                start_time = time.time()

                content_similarity(surfaces[:10])

                elapsed_evaluation_time = time.time() - start_time

                logger.info(
                    "{} evaluation took {:07.3f}s ({:09.3f} slower than dim_size={:02d})".format(
                        similarity_name,
                        elapsed_evaluation_time,
                        elapsed_evaluation_time / previously_elapsed_training_time,
                        dim_size // 2,
                    ),
                )

                df = df.append(
                    {
                        "m_similarity": similarity_name,
                        "n_surfaces": number_of_surfaces,
                        "s_dim": dim_size,
                        "e_training_time": elapsed_training_time,
                        "e_evaluation_time": elapsed_evaluation_time,
                        "e_training_time_ratio": elapsed_training_time
                        / previously_elapsed_training_time,
                        "e_evaluation_time_ratio": elapsed_evaluation_time
                        / previously_elapsed_evaluation_time,
                    },
                    ignore_index=True,
                )

                previously_elapsed_training_time = elapsed_training_time
                previously_elapsed_evaluation_time = elapsed_evaluation_time

    if similarity is not None:
        df.to_csv(f"benchmark_{similarity_name}.csv", index=False)
    else:
        df.to_csv("benchmark.csv", index=False)


if __name__ == "__main__":
    benchmark()
