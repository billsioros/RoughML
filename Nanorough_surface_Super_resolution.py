# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
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

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="e24f0051" outputId="30ae6242-7890-45fb-97f4-7a3f98101b5c"
GDRIVE_DIR = BASE_DIR / "drive"

try:
    from google.colab import drive

    drive.mount(f"{GDRIVE_DIR}")
except ImportError:
    pass

# +
SECRETS_DIR = GDRIVE_DIR / "MyDrive" / "Secrets"

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

WHEEL_VERSION = "3.0.1"
WHEEL_FILE = "roughml-%s-py3-none-any.whl" % (WHEEL_VERSION,)
WHEEL_PATH = THESIS_DIR / "Binaries" / WHEEL_FILE

import os
import random

# + cellView="code" colab={"base_uri": "https://localhost:8080/"} id="1057687b" outputId="2ab1f525-0235-4308-cabb-a7793277473b"
import subprocess
import sys

import numpy as np

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
# + cellView="code" id="4d6c30c9"
import torch

SEED = 1234

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

# + cellView="code" id="520ba5c1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -


# ## Setting up our callbacks

# ### End-of-training callback

# +
from datetime import datetime

from roughml.shared.notifiers import EndOfTrainingNotifier

training_callback = None
if SECRETS_DIR.is_dir():
    notifier = EndOfTrainingNotifier.from_json(SECRETS_DIR / "credentials.json")

    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")

    def training_callback(log_file=None, **context):
        notifier(
            ("Vasilis Sioros", "billsioros97@gmail.com"),
            log_file=log_file,
            dataset=context["dataset"],
            generator=context["generator"],
            discriminator=context["discriminator"],
            elapsed_time=context["elapsed_time"],
            succeeded=context["succeeded"],
            identifier=timestamp,
        )


# -

# ### Logging initialization callback


def logging_callback(config, logging_dir):
    level = config.handlers.file.level.lower()

    config.handlers.file.filename = logging_dir / f"{level}.log"

    return config


# + [markdown] id="60d78d1e"
# # 🙃 A naive-approach

# + [markdown] id="ec374650"
# ## Defining the **Generator** and the **Discriminator** instantiation callbacks

# + cellView="code" id="d6594cb1"
from roughml.models import PerceptronGenerator


def get_generator():
    return PerceptronGenerator.from_device(device)


# + cellView="code" id="cac059ee"
from roughml.models import PerceptronDiscriminator


def get_discriminator(generator):
    return PerceptronDiscriminator.from_generator(generator)


# + [markdown] id="eb813599"
# ## Training

# + cellView="code" id="8a131c21"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

# + cellView="code" id="63114157"
import functools

from torch.optim import Adam

from roughml.content.loss import NGramGraphContentLoss
from roughml.data.loaders import load_multiple_datasets_from_pt
from roughml.data.transforms import To, View
from roughml.training.epoch import per_epoch
from roughml.training.flow import TrainingFlow

training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training={
        "manager": {
            "benchmark": True,
            # Uncomment if you want to enable checkpointing
            # "checkpoint": {"multiple": True},
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
        # Uncomment if you want to enable checkpointing
        # "cache": "n_gram_graph_content_loss.pkl",
    },
    data={
        "loader": functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
            transforms=[To(device), View(1, 128, 128)],
            limit=(2, 10),
        )
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

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="836ed418" outputId="f4d7ef3c-027c-4725-9f07-50e4d7c28ff1" tags=[]
training_flow(get_generator, get_discriminator)

# + [markdown] id="fe589c1a"
# # 😎 A CNN based approach

# + [markdown] id="b16a4946"
# ## Instantiating the **Generator** and the **Discriminator** Networks

# + cellView="code" id="e301a7a0"
from roughml.models import CNNGenerator


def get_generator():
    return CNNGenerator.from_device(device)


# + cellView="code" id="5a8a9aad"
from roughml.models import CNNDiscriminator


def get_discriminator(generator):
    return CNNDiscriminator.from_generator(generator)


# + [markdown] id="7bbbcc22"
# ## Training

# + cellView="code" id="45778b4c"
from torch.nn import BCELoss

criterion = BCELoss().to(device)

# + cellView="code" id="82fb12f6"
import functools

from torch.optim import Adam

from roughml.data.transforms import To, View
from roughml.training.epoch import per_epoch

training_flow = TrainingFlow(
    output_dir=OUTPUT_DIR,
    logging={"config": LOGGING_CONFIG, "callback": logging_callback},
    training={
        "manager": {
            "benchmark": True,
            # Uncomment if you want to enable checkpointing
            # "checkpoint": {"multiple": True},
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
        "callbacks": [
            training_callback,
        ],
    },
    content_loss={
        "type": NGramGraphContentLoss,
        # Uncomment if you want to enable checkpointing
        # "cache": "n_gram_graph_content_loss.pkl",
    },
    data={
        "loader": functools.partial(
            load_multiple_datasets_from_pt,
            DATASET_DIR,
            transforms=[To(device), View(1, 128, 128)],
            limit=(2, 10),
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
    suppress_exceptions=False,
)

# + colab={"base_uri": "https://localhost:8080/", "height": 573} id="3c4cdce1" outputId="c22dfcc0-c6d9-4726-c197-acf11bdec52f"
training_flow(get_generator, get_discriminator)
# -

# # 👋 Dismounting Google Drive and persisting any changes made

try:
    from google.colab import drive

    drive.flush_and_unmount()
except ImportError:
    pass
