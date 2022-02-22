import json
import logging
import os
import pprint
import time

import numpy as np
import torch
from data.concept_property_dataset import ConceptPropertyDataset, TestDataset

# from data.concept_property_test_dataset import TestDataset
from model.concept_property_model import ConceptPropertyModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_logger(config):

    log_file_name = f"logs/100kDataExp/{config.get('experiment_name')}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.log"
    print("config.get('experiment_name') :", config.get("experiment_name"))
    print("\n log_file_name :", log_file_name)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(levelname)s : %(name)s - %(message)s",
    )


log = logging.getLogger(__name__)


def read_config(config_file):

    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            config_dict = json.load(json_file)
            set_logger(config_dict)
            return config_dict
    else:
        return config_file


def create_dataset_and_dataloader(dataset_params, dataset_type):

    if dataset_type in ("train", "valid"):
        dataset = ConceptPropertyDataset(dataset_params, dataset_type)
        data_sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    elif dataset_type == "test":

        dataset = TestDataset(dataset_params)
        data_sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=dataset_params["loader_params"]["batch_size"],
            sampler=data_sampler,
            num_workers=dataset_params["loader_params"]["num_workers"],
            pin_memory=dataset_params["loader_params"]["pin_memory"],
        )

    return dataset, dataloader


def create_model(model_params):
    return ConceptPropertyModel(model_params)


def compute_scores(labels, preds):

    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

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


def calculate_loss(
    dataset, batch, concept_embedding, property_embedding, loss_fn, device
):

    # self.concept2idx, self.idx2concept = self.create_concept_idx_dicts()
    # self.property2idx, self.idx2property = self.create_property_idx_dicts()

    # print ("con_pro_dict :", dataset.con_pro_dict, "\n")

    # print ("\t  num_neg_concept :", num_neg_concept, flush=True)

    batch_logits, batch_labels = [], []

    concept_id_list_for_batch = torch.tensor(
        [dataset.concept2idx[concept] for concept in batch[0]], device=device
    )
    property_id_list_for_batch = torch.tensor(
        [dataset.property2idx[prop] for prop in batch[1]], device=device
    )

    # print ("concept_id_list_for_batch :", concept_id_list_for_batch)
    # print ("property_id_list_for_batch :", property_id_list_for_batch)

    # neg_concept_list, neg_property_list = [], []

    logits_pos_concepts = (
        (concept_embedding * property_embedding)
        .sum(-1)
        .reshape(concept_embedding.shape[0], 1)
    )
    labels_pos_concepts = torch.ones_like(
        logits_pos_concepts, dtype=torch.float32, device=device
    )

    batch_logits.append(logits_pos_concepts.flatten())
    batch_labels.append(labels_pos_concepts.flatten())

    # print ("\nlogits_pos_concepts :", logits_pos_concepts)
    # print ("labels :", labels)

    loss_pos_concept = loss_fn(logits_pos_concepts, labels_pos_concepts)
    # print ("Loss positive concepts :", loss_pos_concept)

    loss_neg_concept = 0.0

    for i in range(len(concept_id_list_for_batch)):

        concept_id = concept_id_list_for_batch[i]

        # Extracting the property of the concept at the whole dataset level.
        property_id_list_for_concept = torch.tensor(
            dataset.con_pro_dict[concept_id.item()], device=device
        )

        # Extracting the negative property by excluding the properties that the concept may have at the  whole dataset level
        negative_property_id_for_concept = torch.tensor(
            [
                x
                for x in property_id_list_for_batch
                if x not in property_id_list_for_concept
            ],
            device=device,
        )

        positive_property_for_concept_mask = torch.tensor(
            [
                [1] if x in negative_property_id_for_concept else [0]
                for x in property_id_list_for_batch
            ],
            device=device,
        )

        neg_property_embedding = torch.mul(
            property_embedding, positive_property_for_concept_mask
        )

        concept_i_repeated = (
            concept_embedding[i].unsqueeze(0).repeat(concept_embedding.shape[0], 1)
        )

        logits_neg_concepts = (
            (concept_i_repeated * neg_property_embedding)
            .sum(-1)
            .reshape(concept_i_repeated.shape[0], 1)
        )

        labels_neg_concepts = torch.zeros_like(
            logits_neg_concepts, dtype=torch.float32, device=device
        )

        batch_logits.append(logits_neg_concepts.flatten())
        batch_labels.append(labels_neg_concepts.flatten())

        loss_neg_concept += loss_fn(logits_neg_concepts, labels_neg_concepts)

    batch_logits = torch.vstack(batch_logits).reshape(-1, 1)
    batch_labels = torch.vstack(batch_labels).reshape(-1, 1)

    return loss_pos_concept + loss_neg_concept, batch_logits, batch_labels

