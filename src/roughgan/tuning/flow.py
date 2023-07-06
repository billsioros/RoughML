from ray import tune

from roughgan.shared.configuration import Configuration
from roughgan.tuning.manager import TuningManager


class TuningFlow(Configuration):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not hasattr(self.parameters, "benchmark"):
            self.parameters.benchmark = True

        if not hasattr(self.parameters, "log_every_n"):
            self.parameters.log_every_n = 10

        if not hasattr(self.parameters, "n_epochs"):
            self.parameters.n_epochs = 20

        if not hasattr(self.parameters, "train_ratio"):
            self.parameters.train_ratio = 0.8

        if not hasattr(self.parameters.optimizer, "params"):
            self.parameters.optimizer.params = {
                "lr": tune.loguniform(1e-4, 1e-1),
                "betas": (0.5, 0.999),
                "weight_decay": tune.uniform(0, 1e-4),
            }

        if not hasattr(self.parameters, "dataloader"):
            self.parameters.dataloader = {
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
            }

    def __call__(self, get_generator, get_discriminator, **training_manager_config):
        self._tuning_manager = TuningManager(
            get_generator=get_generator,
            get_discriminator=get_discriminator,
            **training_manager_config,
        )

        self._tuning_manager(
            self.trial_factory,
            self.dataloader,
            **self.parameters.to_dict(),
        )

    @property
    def analysis(self):
        return self._tuning_manager.analysis

    @property
    def dataframe(self):
        return self._tuning_manager.dataframe

    @property
    def best_trial(self):
        return self._tuning_manager.best_trial

    @property
    def best_models(self):
        return self._tuning_manager.best_models
