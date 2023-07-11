from pathlib import Path

BASE_DIR = Path.cwd()
GDRIVE_DIR = BASE_DIR / "drive"

try:
    from google.colab import drive

    drive.mount(f"{GDRIVE_DIR}")
except ImportError:
    pass
SECRETS_DIR = GDRIVE_DIR / "MyDrive" / "Secrets"

THESIS_DIR = GDRIVE_DIR / "MyDrive" / "Thesis" if GDRIVE_DIR.is_dir() else BASE_DIR

OUTPUT_DIR = THESIS_DIR / "Output"

DATASET_DIR = THESIS_DIR / "data" if THESIS_DIR.is_dir() else BASE_DIR / "data"
import os

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "CRITICAL").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"standard": {"format": "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"}},
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

WHEEL_VERSION = "3.0.1"
WHEEL_FILE = f"roughgan-{WHEEL_VERSION}-py3-none-any.whl"
WHEEL_PATH = THESIS_DIR / "Binaries" / WHEEL_FILE

import os
import random
import subprocess
import sys

import numpy as np

pip_freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()

if "roughgan" not in pip_freeze_output:
    if WHEEL_PATH.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", WHEEL_PATH])
    else:
        raise FileNotFoundError(WHEEL_PATH)
import torch

SEED = 1234

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


training_callback = None


def logging_callback(config, logging_dir):
    level = config.handlers.file.level.lower()

    config.handlers.file.filename = logging_dir / f"{level}.log"

    return config


from roughgan.models import PerceptronGenerator


def get_generator():
    return PerceptronGenerator.from_device(device)


from roughgan.models import PerceptronDiscriminator


def get_discriminator(generator):
    return PerceptronDiscriminator.from_generator(generator)


from torch.nn import BCELoss

criterion = BCELoss().to(device)
import functools

from torch.optim import Adam

from roughgan.content.loss import NGramGraphContentLoss
from roughgan.data.loaders import load_multiple_datasets_from_pt
from roughgan.data.transforms import To, View
from roughgan.training.epoch import per_epoch
from roughgan.training.flow import TrainingFlow

training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training={
        "manager": {
            "benchmark": True,
            "train_epoch": per_epoch,
            "log_every_n": 10,
            "criterion": {"instance": criterion},
            "n_epochs": 10,
            "train_ratio": 0.8,
            "optimizer": {
                "type": Adam,
                "params": {"lr": 0.1, "weight_decay": 0},
            },
            "dataloader": {
                "batch_size": 256,
                "shuffle": True,
                "num_workers": 0,
            },
        },
        "callbacks": [
            training_callback,
        ],
    },
    content_loss={
        "type": NGramGraphContentLoss,
    },
    data={
        "loader": functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
            transforms=[To(device), View(1, 128, 128)],
            limit=(2, 10),
        ),
    },
    animation={
        "indices": [
            0,
        ],
        "save_path": "perceptron_per_epoch_animation.mp4",
    },
    plot={
        "grayscale": {"limit": 10, "save_path_fmt": "grayscale/%s_%02d.png"},
        "surface": {"limit": 10, "save_path_fmt": "surface/%s_%02d.png"},
        "against": {"save_path_fmt": "against_%s.png"},
    },
    suppress_exceptions=False,
)
from roughgan.models import CNNGenerator


def get_generator():
    return CNNGenerator.from_device(device)


from roughgan.models import CNNDiscriminator


def get_discriminator(generator):
    return CNNDiscriminator.from_generator(generator)


from torch.nn import BCELoss

criterion = BCELoss().to(device)
import functools

from torch.optim import Adam

from roughgan.content.loss import ArrayGraph2DContentLoss
from roughgan.data.transforms import To, View
from roughgan.training.epoch import per_epoch

training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training={
        "manager": {
            "benchmark": True,
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
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
            },
        },
        "callbacks": [
            training_callback,
        ],
    },
    content_loss={
        "type": ArrayGraph2DContentLoss,
    },
    data={
        "loader": functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
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

try:
    from google.colab import drive

    drive.flush_and_unmount()
except ImportError:
    pass
