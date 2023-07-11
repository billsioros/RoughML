from abc import ABC, abstractmethod

import torch


class Transform(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Flatten(Transform):
    def __call__(self, tensor):
        return torch.flatten(tensor)


class To(Transform):
    def __init__(self, device) -> None:
        self.device = device

    def __call__(self, tensor):
        return tensor.to(self.device)


class Normalize(Transform):
    def callback(self, dataset):
        self.min = torch.min(dataset.surfaces).item()
        self.max = torch.max(dataset.surfaces).item()

    def __call__(self, tensor):
        if self.max - self.min > 0:
            return (tensor - self.min) / (self.max - self.min)

        return torch.zeros(tensor.size())


class View(Transform):
    def __init__(self, *args) -> None:
        self.args = args

    def __call__(self, tensor):
        return tensor.view(*self.args)
