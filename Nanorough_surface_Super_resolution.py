# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: ''
#     name: ''
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/billsioros/thesis/blob/master/Nanorough_surface_Super_resolution.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="cce26209"
# # ✔️ Prerequisites

# + [markdown] id="affab565"
# First of all we need to take care of a few **prerequisites**, most notably:
#
# - Install the various pip modules that we will be using.
# - Install some linux specific dependencies of our [content loss](#content-loss).
# - Initialize the Random Number Generator(s), so that our experiments can be replicated.
# - Determine:
#   - The current working directory, as it's going to be used to reference various files such as the dataset, our model checkpoints e.t.c
#   - The available hardware backend. GPU utilization is preferable, as it results in higher complition time.
# - `(Optionally)` Mount Google Drive, where we can load our dataset from.

# + [markdown] id="19e0a6d0"
# ## Determining the Current Working Directory

# + cellView="code" id="945a9ccd"
from pathlib import Path

BASE_DIR = Path.cwd()

# + [markdown] id="94c4d99f"
# ## Mounting Google Drive

# + cellView="code" id="12dcaea0"
GDRIVE_DIR = BASE_DIR / "drive"

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="e24f0051" outputId="30ae6242-7890-45fb-97f4-7a3f98101b5c"
try:
    from google.colab import drive

    drive.mount(f"{GDRIVE_DIR}")
except ImportError:
    pass

# +
if GDRIVE_DIR.is_dir():
    THESIS_DIR = GDRIVE_DIR / "MyDrive" / "Thesis"
else:
    THESIS_DIR = BASE_DIR

OUTPUT_DIR = THESIS_DIR / "Output"

if THESIS_DIR.is_dir():
    DATASET_DIR = THESIS_DIR / "Datasets"
else:
    DATASET_DIR = BASE_DIR / "Datasets"
# -

# ## Configuring our Loggers

# +
import os

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "CRITICAL").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"}
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

# + [markdown] id="16a902e2"
# ## Installing [graphviz](https://graphviz.org/) & [libgraphviz-dev](https://packages.debian.org/jessie/libgraphviz-dev)

# + [markdown] id="8c0cdd85"
# The aforementioned packages are required by [PyINSECT](https://github.com/billsioros/PyINSECT/tree/implementing-HPGs) and more specifically its graph plotting methods.

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="919734ca" outputId="0767e6fd-b55e-4ebc-9e0e-11b243700c1c"
# !sudo apt-get install graphviz libgraphviz-dev 1> /dev/null

# + [markdown] id="7f5668f4"
# ## Installing the required `pip` modules

# + [markdown] id="aebad6f1"
# - [torch](https://pytorch.org/) is our machine learning framework of choice.
# - [numpy](https://numpy.org/), [sympy](https://www.sympy.org/en/index.html) and [scipy](https://www.scipy.org/) are used to in the context of nanorough surface generation.
# - [plotly](https://plotly.com/) (which requires [pandas](https://pandas.pydata.org/)) as well as [matplotlib](https://matplotlib.org/) are used in order to plot various graphs.
# -

WHEEL_VERSION = "2.1.0"
WHEEL_FILE = "roughml-%s-py3-none-any.whl" % (WHEEL_VERSION,)
WHEEL_PATH = THESIS_DIR / "Binaries" / WHEEL_FILE

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="1057687b" outputId="2ab1f525-0235-4308-cabb-a7793277473b"
import subprocess
import sys

pip_freeze_output = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze"]
).decode()

if "roughml" not in pip_freeze_output:
    if WHEEL_PATH.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", WHEEL_PATH])
    else:
        raise FileNotFoundError(WHEEL_PATH)

# + [markdown] id="0192c059"
# ## Initializing (a.k.a `Seeding`) the Random Number Generator(s)

# + [markdown] id="fa44f756"
# We are required to seed various random number generation engines, so that our experiments can be replicated on a later date.

# + cellView="code" id="5daee8da"
SEED = 1234

import os
import random

import numpy as np

# + cellView="code" id="4d6c30c9"
import torch

if SEED is not None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)

# + [markdown] id="6fce8a2a"
# ## Determining available backend

# + [markdown] id="5ca328f8"
# By default, we are going to be utilizing the available CPU backend, if no GPU is available.

# + cellView="code" id="ebabd4a5"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

# + cellView="code" id="520ba5c1"
device = torch.device(device)

# + [markdown] id="60d78d1e"
# # 🙃 A naive-approach

# + [markdown] id="ec374650"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="d6594cb1"
from roughml.models import PerceptronGenerator

generator = PerceptronGenerator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="4bff3a44" outputId="9d75bc1a-7272-4fdb-902d-e689a0627517"
generator

# + cellView="code" id="cac059ee"
from roughml.models import PerceptronDiscriminator

discriminator = PerceptronDiscriminator.from_generator(generator)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="64022987" outputId="7adfec62-6485-49aa-c5e9-667d291b03e9"
discriminator

# + [markdown] id="eb813599"
# ## Training

# + cellView="code" id="8a131c21"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

import functools

# + cellView="code" id="63114157"
from roughml.content.loss import NGramGraphContentLoss
from roughml.data.loaders import (
    load_dataset_from_pt,
    load_dataset_from_zip,
    load_multiple_datasets_from_pt,
)
from roughml.data.transforms import To, View
from roughml.training.flow import TrainingFlow
from roughml.training.manager import per_epoch


def logging_callback(config, logging_dir):
    level = config["handlers"]["file"]["level"].lower()

    config["handlers"]["file"]["filename"] = logging_dir / f"{level}.log"

    return config


training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training_manager={
        "benchmark": True,
        "checkpoint": {"multiple": True},
        "train_epoch": per_epoch,
        "log_every_n": 10,
        "criterion": {"instance": criterion},
        "n_epochs": 10,
        "train_ratio": 0.8,
        "optimizer": {"lr": 0.0005, "weight_decay": 0},
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
        },
    },
    content_loss={
        "type": NGramGraphContentLoss,
        "cache": "n_gram_graph_content_loss.pkl",
    },
    data={
        "loader": functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
            transforms=[To(device), View(1, 128, 128)],
            limit=10,
        )
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
)

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="836ed418" outputId="f4d7ef3c-027c-4725-9f07-50e4d7c28ff1" tags=[]
training_flow(generator, discriminator)

# + [markdown] id="fe589c1a"
# # 😎 A CNN based approach

# + [markdown] id="b16a4946"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="e301a7a0"
from roughml.models import CNNGenerator

generator = CNNGenerator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="36ffe487" outputId="eb6bbc54-4dd6-48f7-bf24-20becc244a3a"
generator

# + cellView="code" id="5a8a9aad"
from roughml.models import CNNDiscriminator

discriminator = CNNDiscriminator.from_device(device)

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="1d0a4a40" outputId="2c54123a-5dce-4405-a82e-a28c832e678e"
discriminator

# + [markdown] id="7bbbcc22"
# ## Training

# + cellView="code" id="45778b4c"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

import functools

# + cellView="code" id="82fb12f6"
from roughml.content.loss import ArrayGraph2DContentLoss
from roughml.data.transforms import To, View
from roughml.training.flow import TrainingFlow
from roughml.training.manager import per_epoch

training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training_manager={
        "benchmark": True,
        "checkpoint": {"multiple": True},
        "train_epoch": per_epoch,
        "log_every_n": 10,
        "criterion": {"instance": criterion},
        "n_epochs": 10,
        "train_ratio": 0.8,
        "optimizer": {"lr": 0.0002, "betas": (0.5, 0.999)},
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 0,
        },
    },
    content_loss={
        "type": ArrayGraph2DContentLoss,
        "cache": "array_graph2d_content_loss.pkl",
    },
    data={
        functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
            transforms=[To(device), View(1, 128, 128)],
            limit=10,
        )
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
)

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="3c4cdce1" outputId="c22dfcc0-c6d9-4726-c197-acf11bdec52f"
training_flow(generator, discriminator)
# -

# ## Dismounting Google Drive and persisting any changes made

try:
    from google.colab import drive

    drive.flush_and_unmount()
except ImportError:
    pass
