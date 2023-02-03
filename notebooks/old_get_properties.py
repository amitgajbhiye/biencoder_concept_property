import argparse
from fileinput import filename
import logging
import os
import pickle
import pandas as pd

import torch

from utils.functions import (
    create_model,
    read_config,
    to_cpu,
)

from get_embedding import generate_embedings

log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda") if cuda_available else torch.device("cpu")


def get_property_embeddings(config):

    generate_embedings(config=config)


if __name__ == "__main__":

    log.info(f"\n {'*' * 50}")
    log.info(f"Generating the Concept Property Embeddings")

    parser = argparse.ArgumentParser(description="Concept Property Biencoder Model")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The embeddings are generated with the following configuration")

    log.info(f"\n {config} \n")

    get_property_embeddings(config=config)

