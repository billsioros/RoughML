import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class NanoroughSurfaceDataset(Dataset):
    """A dataset of pre-generated nanorough surfaces."""

    def __init__(self, surfaces, subsampling_factor, transforms) -> None:
        self.surfaces = surfaces

        self.subsampling_factor = subsampling_factor
        self.subsampling_value = int(self.surfaces.shape[2] / subsampling_factor)

        self.transforms = transforms

        for transform in self.transforms:
            if hasattr(transform, "callback"):
                transform.callback(self)

    def __len__(self) -> int:
        return len(self.surfaces)

    def __getitem__(self, idx):
        surface = self.surfaces[idx]

        for transform in self.transforms:
            surface = transform(surface)

        return surface

    def to_pt(self, path):
        state = {
            "surfaces": self.surfaces,
            "subsampling_factor": self.subsampling_factor,
            "transforms": self.transforms,
        }

        torch.save(state, path)

    @classmethod
    def from_list(cls, surfaces, subsampling_factor=4, transforms=[]):
        surfaces = np.array(surfaces)
        surfaces = torch.from_numpy(surfaces)

        return cls(surfaces, subsampling_factor, transforms)

    @classmethod
    def from_pt(cls, path):
        state = torch.load(path)

        return cls(state["surfaces"], state["subsampling_factor"], state["transforms"])
