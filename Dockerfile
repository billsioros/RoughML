FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04 as cuda-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

FROM cuda-base as pyenv-base

ENV PYENV_ROOT=/opt/.pyenv \
    PYENV_GIT_TAG=v2.2.5

ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

RUN curl https://pyenv.run | bash

FROM pyenv-base as python-base

COPY --from=pyenv-base ${PYENV_ROOT} ${PYENV_ROOT}

ENV PYTHON_VERSION=3.9.4 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash

FROM python-base as poetry-base

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_VERSION=1.2.2

ENV POETRY_CACHE_DIR=${POETRY_HOME}/.cache \
    PATH=${POETRY_HOME}/bin:${PATH}

COPY --from=python-base ${PYENV_ROOT} ${PYENV_ROOT}

RUN curl -sSL https://install.python-poetry.org | python -

RUN mkdir -p ${POETRY_CACHE_DIR}/virtualenvs

FROM poetry-base as build-base

ENV USER=app \
    GROUP=app

ENV WORKSPACE=/home/${USER}/app

RUN adduser ${USER} && \
    chown -R ${USER}:${GROUP} /home/${USER}

COPY --from=poetry-base ${PYENV_ROOT} ${PYENV_ROOT}
COPY --from=poetry-base ${POETRY_HOME} ${POETRY_HOME}

RUN mkdir -p ${WORKSPACE}

WORKDIR ${WORKSPACE}

COPY pyproject.toml poetry.lock ${WORKSPACE}

RUN poetry env use ${PYTHON_VERSION} && \
    poetry run pip install --upgrade pip && \
    poetry install --without dev

FROM build-base as build

COPY --from=build-base ${WORKSPACE} ${WORKSPACE}

COPY src ${WORKSPACE}

USER ${USER}

CMD poetry run python roughgan/model.py train \
    --dataset data \
    --output output
