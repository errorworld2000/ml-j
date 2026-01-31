import click
import yaml
from rich.console import Console

from mini_ml.core.trainer import Trainer
from mini_ml.utils.config import AppConfig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "-c",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the training configuration YAML file.",
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir_override",
    type=click.Path(resolve_path=True),
    default=None,
    help="Override the output directory specified in the config file.",
)
def main(config_path: str, output_dir_override: str):
    """
    A reproducible deep learning model training script based on configuration files.

    Examples:

    python train.py -c configs/my_experiment.yaml

    python train.py -c configs/my_experiment.yaml -o outputs/new_run
    """
    # 1. Load YAML configuration
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 2. Override output directory if provided
    if output_dir_override:
        if "environment" not in raw_config:
            raw_config["environment"] = {}
        raw_config["environment"]["output_dir"] = output_dir_override

    # 3. Validate configuration with Pydantic
    try:
        config = AppConfig(**raw_config)
    except Exception:
        console = Console()
        console.print("❌ [bold red]Error: Configuration validation failed![/bold red]")
        console.print(
            "Please check your YAML file against the Pydantic models in config.py."
        )
        console.print_exception(show_locals=True)
        return

    # 4. Initialize and run Trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
