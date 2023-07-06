<h1 align="center">RoughGAN</h1>

<p align="center">
  <a href="https://github.com/billsioros/RoughGAN/releases">
    <img
      src="https://img.shields.io/github/v/release/billsioros/RoughGAN"
      alt="Latest Release"
    />
  </a>
  <!-- TODO -->
  <!-- <a href="https://github.com/billsioros/RoughGAN/actions/workflows/ci.yml">
    <img
      src="https://github.com/billsioros/RoughGAN/actions/workflows/ci.yml/badge.svg"
      alt="CI"
    />
  </a> -->
  <a href="https://github.com/billsioros/RoughGAN/actions/workflows/cd.yml">
    <img
      src="https://github.com/billsioros/RoughGAN/actions/workflows/cd.yml/badge.svg"
      alt="CD"
    />
  </a>
  <a href="https://results.pre-commit.ci/latest/github/billsioros/RoughGAN/master">
    <img
      src="https://results.pre-commit.ci/badge/github/billsioros/RoughGAN/master.svg"
      alt="pre-commit.ci status"
    />
  </a>
  <a href="https://app.renovatebot.com/dashboard#github/billsioros/RoughGAN">
    <img
      src="https://img.shields.io/badge/renovate-enabled-brightgreen.svg?style=flat&logo=renovatebot"
      alt="Renovate - Enabled">
  </a>
  <!-- TODO -->
  <!-- <a href="https://codecov.io/gh/billsioros/RoughGAN">
    <img
      src="https://codecov.io/gh/billsioros/RoughGAN/branch/master/graph/badge.svg?token=coLOL0j6Ap"
      alt="Test Coverage"/>
  </a> -->
  <a href="https://opensource.org/licenses/MIT">
    <img
      src="https://img.shields.io/badge/license-MIT-green"
      alt="License"
    />
  </a>
  <a target="_blank" href="https://colab.research.google.com/github/billsioros/RoughML">
    <img
      src="https://colab.research.google.com/assets/colab-badge.svg"
      alt="Open In Colab"/>
  </a>
  <a href="https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/billsioros/RoughGAN">
    <img
      src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode"
      alt="Open in GitHub Codespaces"
    />
  </a>
  <a href="https://github.com/billsioros/cookiecutter-pypackage">
    <img
      src="https://img.shields.io/badge/cookiecutter-template-D4AA00.svg?style=flat&logo=cookiecutter"
      alt="Cookiecutter Template">
  </a>
  <a href="https://www.buymeacoffee.com/billsioros">
    <img
      src="https://img.shields.io/badge/Buy%20me%20a-coffee-FFDD00.svg?style=flat&logo=buymeacoffee"
      alt="Buy me a coffee">
  </a>
</p>

> Accompanying code for the paper [**Generating Realistic Nanorough Surfaces Using an N-Gram-Graph Augmented Deep Convolutional Generative Adversarial Network**](https://dl.acm.org/doi/fullHtml/10.1145/3549737.3549794) presented at [**SETN 2022**](https://hilab.di.ionio.gr/setn2022/).

In this work, we look at how a Generative Adversarial Network (GAN)-based strategy, given a nanorough surface data set, may learn to produce nanorough surface samples that are statistically equivalent to the ones belonging to the training data set. We also look at how combining the GAN framework with a variety of nanorough similarity measures might improve the realisticity of the synthesized nanorough surfaces. We showcase via multiple experiments that our framework is able to produce sufficiently realistic nanorough surfaces, in many cases indistinguishable from real ones.

## :cd: Getting started

You can run the model locally using the following commands:

```bash
docker build . -t roughgan:$( git tag -l | tail -1 | cut -c2- ) -t build:train -f Dockerfile
docker run -v $(pwd)/data:/home/app/app/data -v $(pwd)/models:/home/app/app/models --gpus $(nvidia-smi --list-gpus | wc -l) roughgan:latest
```

The project's documentation can be found [here](https://billsioros.github.io/RoughGAN/).

## :heart: Support the project

If you would like to contribute to the project, please go through the [Contributing Guidelines](https://billsioros.github.io/RoughGAN/latest/CONTRIBUTING/) first.

You can also support the project by [**Buying me a coffee! ☕**](https://www.buymeacoffee.com/billsioros).

## :bookmark_tabs: Citation

```bibtex
@inproceedings{10.1145/3549737.3549794,
  author = {Sioros, Vasilis and Giannakopoulos, George and Constantoudis, Vassileios},
  title = {Generating Realistic Nanorough Surfaces Using an N-Gram-Graph Augmented Deep Convolutional Generative Adversarial Network},
  year = {2022},
  isbn = {9781450395977},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3549737.3549794},
  doi = {10.1145/3549737.3549794},
  booktitle = {Proceedings of the 12th Hellenic Conference on Artificial Intelligence},
  articleno = {53},
  numpages = {10},
  keywords = {Machine Learning, Rough Surfaces, Graph Theory, Nanotechnology, Artificial Intelligence},
  location = {Corfu, Greece},
  series = {SETN '22}
}
```

This project was generated with [`billsioros/cookiecutter-pypackage`](https://github.com/billsioros/cookiecutter-pypackage) cookiecutter template.
