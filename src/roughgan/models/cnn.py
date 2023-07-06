import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

from roughgan.models.base import Base


class CNNBase(Base):
    @classmethod
    def initialize_weights(cls, model):
        classname = model.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    @classmethod
    def from_device(cls, device, *args, dtype=torch.float64, gradient_clipping=None, **kwargs):
        model = super().from_device(
            device,
            *args,
            dtype=dtype,
            gradient_clipping=gradient_clipping,
            **kwargs,
        )

        model.apply(cls.initialize_weights)

        return model


class CNNGenerator(Base):
    def __init__(
        self,
        in_channels=100,
        out_channels=128,
        training_channels=1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training_channels = training_channels

        self.feature_dims = (in_channels, 1, 1)

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels * 16, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels * 16),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 16, out_channels * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 8),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 8, out_channels * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(out_channels, training_channels, 4, 2, 1, bias=False),
                ),
            ],
        )

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        for i, module in enumerate(self.module_list):
            batch = module(batch)

            logger.debug(
                "%s:%s #%02d: %s",
                self.__class__.__name__,
                module.__class__.__name__,
                i + 1,
                batch.shape,
            )

        return batch


class CNNDiscriminator(Base):
    def __init__(self, out_channels=1, in_channels=128) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, in_channels, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels * 4, in_channels * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(in_channels * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(nn.Conv2d(in_channels * 8, 1, 8, 1, 0, bias=False), nn.Sigmoid()),
            ],
        )

    def forward(self, batch):
        logger.debug("%s:Input: %s", self.__class__.__name__, batch.shape)

        for i, module in enumerate(self.module_list):
            batch = module(batch)

            logger.debug(
                "%s:%s #%02d: %s",
                self.__class__.__name__,
                module.__class__.__name__,
                i + 1,
                batch.shape,
            )

        return batch

    @classmethod
    def from_generator(cls, generator, dtype=torch.float64, gradient_clipping=None):
        return cls.from_device(
            generator.device,
            dtype=dtype,
            gradient_clipping=gradient_clipping,
        )
