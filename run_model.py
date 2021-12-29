import logging
import argparse
from utils.functions import read_config, display

log = logging.getLogger(__name__)


def train(config):

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Siamese Concept Property Model")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The model is run with the following configuration")
    display(item=config)

    train(config)
