import logging

import torch
from torch import nn

from roughgan.models.base import Base

logger = logging.getLogger(__name__)


class PerceptronGenerator(Base):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features, self.out_features = in_features, out_features

        self.feature_dims = (in_features, 1, 1)

        self.linear = nn.Linear(in_features, out_features[0] * out_features[1])

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        batch_size, _, _, _ = batch.shape

        batch = self.linear(batch.view(batch_size, -1))

        logger.debug("%s:Output: %s", self.__class__.__name__, batch.shape)

        return batch.view(batch_size, 1, self.out_features[0], self.out_features[1])

    @classmethod
    def from_device(
        cls,
        device,
        in_features=100,
        out_features=(128, 128),
        dtype=torch.float64,
        gradient_clipping=None,
    ):
        return super().from_device(
            device,
            in_features,
            out_features,
            dtype=dtype,
            gradient_clipping=gradient_clipping,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset,
        device,
        in_features=100,
        dtype=torch.float64,
        gradient_clipping=None,
    ):
        super_resolved_value = dataset.subsampling_factor * dataset.subsampling_value

        out_features = (super_resolved_value, super_resolved_value)

        return cls.from_device(
            device,
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            gradient_clipping=gradient_clipping,
        )


class PerceptronDiscriminator(Base):
    def __init__(self, in_features) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, 1)
        self.activation = nn.Sigmoid()

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        batch_size, _, _, _ = batch.shape

        batch = self.activation(self.linear(batch.view(batch_size, -1)))

        logger.debug("%s:Output: %s", self.__class__.__name__, batch.shape)

        return batch

    @classmethod
    def from_generator(cls, generator, dtype=torch.float64, gradient_clipping=None):
        return cls.from_device(
            generator.device,
            generator.out_features[0] * generator.out_features[1],
            dtype=dtype,
            gradient_clipping=gradient_clipping,
        )
