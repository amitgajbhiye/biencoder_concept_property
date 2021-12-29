import json
import pprint
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/logfile.log",
    filemode="w",
    format="%(asctime)s : %(levelname)s : %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def read_config(config_file):

    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            return json.load(json_file)
    else:
        return config_file


def display(item):

    with open("logs/logfile.log", "a") as out:
        out.write(pprint.pformat(item))

