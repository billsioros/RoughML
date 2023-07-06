import multiprocessing
from pathlib import Path

import torch
from ray import tune

from roughgan.shared.configuration import Configuration


class TuningManager(Configuration):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not hasattr(self, "available_cpus"):
            self.available_cpus = multiprocessing.cpu_count()

        if not hasattr(self, "available_gpus"):
            self.available_gpus = torch.cuda.device_count()

        if not hasattr(self, "metric"):
            self.metric = "loss"

        if not hasattr(self, "mode"):
            self.mode = "min"

        if not hasattr(self, "scope"):
            self.scope = "all"

        if not hasattr(self, "num_samples"):
            self.num_samples = 100

        if not hasattr(self, "keep_checkpoints_num"):
            self.keep_checkpoints_num = 10

        if not hasattr(self, "local_dir"):
            self.local_dir = Path.cwd() / "ray_results"

    def __call__(self, trial_fn, get_dataset, **config):
        _, dataset = next(get_dataset())

        scheduler = tune.schedulers.ASHAScheduler(
            metric=self.metric,
            mode=self.mode,
            max_t=config["n_epochs"],
            grace_period=1,
            reduction_factor=2,
        )

        reporter = tune.CLIReporter(metric_columns=[self.metric, "training_iteration"])

        self._analysis = tune.run(
            trial_fn(self.get_generator, self.get_discriminator, dataset),
            resources_per_trial={
                "cpu": self.available_cpus,
                "gpu": self.available_gpus,
            },
            config=config,
            num_samples=self.num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=str(self.local_dir),
            checkpoint_score_attr=f"min-{self.metric}",
            keep_checkpoints_num=self.keep_checkpoints_num,
        )

    @property
    def analysis(self):
        try:
            return self._analysis
        except AttributeError as attribute_error:
            msg = "No analysis available. Run the manager first"
            raise TypeError(msg) from attribute_error

    @property
    def dataframe(self):
        return self.analysis.dataframe(metric=self.metric, mode=self.mode).sort_values(
            [
                self.metric,
            ],
        )

    @property
    def best_trial(self):
        return self.analysis.get_best_trial(self.metric, self.mode, self.scope)

    @property
    def best_models(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        generator = self.get_generator()
        discriminator = self.get_discriminator(generator)

        map_location = "cuda:%d" % (device.index,) if device.type == "cuda" else "cpu"

        generator_state, discriminator_state = torch.load(
            Path(self.best_trial.checkpoint.value) / "checkpoint",
            map_location=map_location,
        )

        generator.load_state_dict(generator_state)
        discriminator.load_state_dict(discriminator_state)

        return generator.to(device), discriminator.to(device)
