from pathlib import Path

import torch
from ray import tune


def trial_factory(training_manager_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_location = "cuda:%d" % (device.index,) if device.type == "cuda" else "cpu"

    def trial_wrapper(get_generator, get_discriminator, dataset):
        def trial(config, checkpoint_dir=None):
            training_manager = training_manager_type(**config)

            generator = get_generator()
            discriminator = get_discriminator(generator)

            if checkpoint_dir is not None:
                generator_state, discriminator_state = torch.load(
                    Path(checkpoint_dir) / "checkpoint",
                    map_location=map_location,
                )
                generator.load_state_dict(generator_state)
                discriminator.load_state_dict(discriminator_state)

            for epoch, results in enumerate(training_manager(generator, discriminator, dataset)):
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    torch.save(
                        (generator.state_dict(), discriminator.state_dict()),
                        Path(checkpoint_dir) / "checkpoint",
                    )

                tune.report(loss=results[0])

        return trial

    return trial_wrapper
