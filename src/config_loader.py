import hydra


def load_config():
    with hydra.initialize(config_path="config", version_base=None):
        config = hydra.compose(
            config_name="config", overrides=["hydra/job_logging=disabled"]
        )
    return config
