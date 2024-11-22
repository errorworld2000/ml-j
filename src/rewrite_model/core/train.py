import click
from typing import Optional, List
from omegaconf import OmegaConf

@click.command()
@click.option("--config-file", "-c", type=str, default="config.yaml")
def train(config):
    cfg = OmegaConf.load(config)
    print(OmegaConf.to_container(cfg))
    
