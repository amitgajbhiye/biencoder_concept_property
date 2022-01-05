import json
import pprint
import logging

from dataset.concept_property_dataset import ConceptPropertyDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model.concept_property_model import ConceptPropertyModel

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/logfile.log",
    filemode="w",
    format="%(asctime)s : %(levelname)s : %(name)s - %(message)s",
)

log = logging.getLogger(__name__)


def read_config(config_file):

    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            return json.load(json_file)
    else:
        return config_file


def display(item):

    with open("logs/logfile.log", "a") as out:
        out.write(pprint.pformat(item))


def create_dataloader(dataset_params):

    dataset = ConceptPropertyDataset(dataset_params)

    if dataset_params["loader_params"]["sampler"] == "random":
        data_sampler = RandomSampler(dataset)
    elif dataset_params["loader_params"]["sampler"] == "sequential":
        data_sampler = SequentialSampler(dataset)
    else:
        log.error("Specify a valid data sampler name")

    dataloader = DataLoader(
        dataset,
        batch_size=dataset_params["loader_params"]["batch_size"],
        sampler=data_sampler,
        num_workers=dataset_params["loader_params"]["num_workers"],
        pin_memory=dataset_params["loader_params"]["pin_memory"],
    )

    return dataloader


def create_model(model_params):
    return ConceptPropertyModel(model_params)


def compute_scores(labels, preds):

    scores = {
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores
