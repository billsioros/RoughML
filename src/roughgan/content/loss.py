import concurrent
import contextlib
import itertools
import logging
import os
import pickle
import statistics
import time
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import torch
from pyinsect.collector.NGramGraphCollector import (
    ArrayGraph2DCollector,
    HPG2DCollector,
    NGramGraphCollector,
)
from scipy import fft

from roughgan.content.quantization import KBinsDiscretizerQuantizer
from roughgan.shared.configuration import Configuration

logger = logging.getLogger(__name__)


def per_row(method=None, *, expected_ndim=2):
    """The following decorator, given a multidimensional matrix, applies the decorated
    function on every row of the provided matrix and returns a one dimensional matrix,
    consisting of the accumulated return values of all the calls **or** a singular
    value, in case the multidimensional matrix has less than expected dimensions."""

    def wrapper(method):
        @wraps(method)
        def wrapper_wrapper(self, matrix, *args, **kwargs):
            if len(matrix.shape) > expected_ndim:
                return torch.tensor([method(self, row, *args, **kwargs) for row in matrix])

            return method(self, matrix, *args, **kwargs)

        return wrapper_wrapper

    return wrapper if method is None else wrapper(method)


class ContentLoss(Configuration, ABC):
    def __init__(self, skip_quantization=False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.skip_quantization = skip_quantization

        if self.skip_quantization is False:
            self.quantizer = KBinsDiscretizerQuantizer(**kwargs)
            self.surfaces = self.quantizer.surfaces

    def __len__(self) -> int:
        return len(self.surfaces)

    @abstractmethod
    def __call__(self, surface):
        if self.skip_quantization is False:
            return self.quantizer(surface)

        return surface

    def __str__(self) -> str:
        return str({"shape": self.surfaces.shape})

    def to_pickle(self, path):
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, path):
        with path.open("rb") as file:
            instance = pickle.load(file)

            if not isinstance(instance, cls):
                msg = f"{instance!r} is not an instance of {cls.__name__}"
                raise TypeError(msg)

            return instance


class NGramGraphContentLoss(ContentLoss):
    """An `n-gram graph` based content loss."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.surfaces = self.surfaces.reshape(self.surfaces.shape[0], -1)

        self._collector = NGramGraphCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    @per_row(expected_ndim=1)
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface).reshape(-1))


class ArrayGraph2DContentLoss(ContentLoss):
    """A `2D array graph` based content loss."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._collector = ArrayGraph2DCollector()

        for surface in self.surfaces:
            self._collector.add(surface)

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))


class HPG2DContentLoss(ContentLoss):
    """A `Hierarchical Proximity Graph (HPG)` based content loss."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._build_collector()

    def _build_collector(self):
        total_start_time = time.time()

        self._collector = HPG2DCollector()

        logger.debug("Constructing %02d graphs", len(self.surfaces))

        elapsed_time = []
        for index, surface in enumerate(self.surfaces):
            start_time = time.time()
            self._collector.add(surface)
            elapsed_time.append(time.time() - start_time)

            logger.debug("Constructed graph %02d in %07.3fs", index, elapsed_time[-1])

        logger.debug(
            "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
            len(self.surfaces),
            time.time() - total_start_time,
            statistics.mean(elapsed_time),
            statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
        )

    @per_row
    def __call__(self, surface):
        return self._collector.appropriateness_of(super().__call__(surface))


class HPG2DParallelContentLoss(HPG2DContentLoss):
    def _build_collector(self):
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as pool:
            total_start_time = time.time()

            self._collector = HPG2DCollector()

            futures = {}
            for index, surface in enumerate(self.surfaces):
                future = pool.submit(self._collector.add, surface)

                futures[future] = (index, time.time())

            logger.debug("Awaiting %02d jobs", len(self.surfaces))

            elapsed_time = [None] * len(self.surfaces)
            for future in concurrent.futures.as_completed(futures):
                self._collector._add_graph(future.result())

                index, start_time = futures[future]

                elapsed_time[index] = time.time() - start_time

                logger.debug("Job %02d completed after %07.3fs", index, elapsed_time[index])

            logger.debug(
                "Constructed %02d graphs in %07.3fs [%.3f ± %.3f seconds per graph]",
                len(self.surfaces),
                time.time() - total_start_time,
                statistics.mean(elapsed_time),
                statistics.stdev(elapsed_time) if len(elapsed_time) > 1 else 0,
            )


class VectorSpaceContentLoss(ContentLoss):
    """A content loss that represents surfaces as vectors.

    The content loss calculates the histogram and the fourier transform corresponding to
    each provided surface, in order to construct a vector corresponding to that surface.
    It then utilizes conventional vector distance metrics to calculate the loss value.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(skip_quantization=True, **kwargs)

        with contextlib.suppress(AttributeError):
            self.surfaces = self.surfaces.numpy()

        if not hasattr(self, "n_neighbors"):
            self.n_neighbors = len(self.surfaces)

        self.histograms, self.fouriers = [], []
        for surface in self.surfaces:
            self.histograms.append(np.histogram(surface.reshape(-1))[0])
            self.fouriers.append(np.absolute(fft.fft2(surface)))

    @per_row
    def __call__(self, surface):
        (histogram, _), fourier = np.histogram(surface.reshape(-1)), np.absolute(fft.fft2(surface))

        loss = 0
        for _histogram, _fourier in itertools.islice(
            zip(self.histograms, self.fouriers),
            self.n_neighbors,
        ):
            loss += np.sqrt(np.square(np.subtract(histogram, _histogram)).mean()) / (
                self.n_neighbors * 2
            )

            loss += np.sqrt(np.square(np.subtract(fourier, _fourier)).mean()) / (
                self.n_neighbors * 2
            )

        return 1 / (1 + loss)


if __name__ == "__main__":
    SIZE, DIM = 10, 4

    fixed_noise, tensors = torch.rand(DIM, DIM), torch.rand(SIZE, DIM, DIM)

    print("\nTesting 'NGramGraphContentLoss'")
    content_loss = NGramGraphContentLoss(surfaces=tensors)

    content_losses = [content_loss(row.reshape(-1)) for row in tensors]

    print(
        content_loss(fixed_noise.reshape(-1)),
        (min(content_losses), max(content_losses)),
    )

    print("\nTesting 'ArrayGraph2DContentLoss'")
    content_loss = ArrayGraph2DContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i]) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise), (min(content_losses), max(content_losses)))

    print("\nTesting 'HPG2DContentLoss'")
    content_loss = HPG2DContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i]) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise), (min(content_losses), max(content_losses)))

    print("\nTesting 'VectorSpaceContentLoss'")
    content_loss = VectorSpaceContentLoss(surfaces=tensors)

    content_losses = [content_loss(tensors[i].numpy()) for i in range(tensors.shape[0])]

    print(content_loss(fixed_noise.numpy()), (min(content_losses), max(content_losses)))
