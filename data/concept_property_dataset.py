import logging
from os import sep

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


log = logging.getLogger(__name__)


class ConceptPropertyDataset(Dataset):
    def __init__(self, dataset_params, dataset_type):

        if dataset_type == "train":
            self.data_df = pd.read_csv(
                dataset_params.get("train_file_path"),
                sep="\t",
                header=None,
                names=["concept", "property"],
            )
        elif dataset_type == "valid":
            self.data_df = pd.read_csv(
                dataset_params.get("val_file_path"),
                sep="\t",
                header=None,
                names=["concept", "property"],
            )

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(
            dataset_params.get("hf_tokenizer_path")
        )

        self.concept2idx, self.idx2concept = self.create_concept_idx_dicts()
        self.property2idx, self.idx2property = self.create_property_idx_dicts()

        self.con_pro_dict, self.prop_con_dict = self.populate_dict()

        # print ("self.con_pro_dict :", self.con_pro_dict)

        self.context_num = dataset_params.get("context_num")

    def create_concept_idx_dicts(self):

        unique_concepts = self.data_df["concept"].unique()

        item2idx, idx2item = {}, {}

        for idx, item in enumerate(unique_concepts):
            item2idx[item] = idx
            idx2item[idx] = item

        return item2idx, idx2item

    def create_property_idx_dicts(self):

        unique_properties = self.data_df["property"].unique()

        item2idx, idx2item = {}, {}

        for idx, item in enumerate(unique_properties):
            item2idx[item] = idx
            idx2item[idx] = item

        return item2idx, idx2item

    def populate_dict(self):

        concept_property_dict, property_concept_dict = {}, {}

        unique_concepts = self.data_df["concept"].unique()
        unique_properties = self.data_df["property"].unique()

        self.data_df.set_index("concept", inplace=True)

        for concept in unique_concepts:

            concept_id = self.concept2idx[concept]

            property_list = self.data_df.loc[concept].values.flatten()
            property_ids = np.asarray([self.property2idx[x] for x in property_list])

            concept_property_dict[concept_id] = property_ids

        self.data_df.reset_index(inplace=True)

        self.data_df.set_index("property", inplace=True)

        for prop in unique_properties:

            property_id = self.property2idx[prop]

            concept_list = self.data_df.loc[prop].values.flatten()
            concept_ids = np.asarray([self.concept2idx[x] for x in concept_list])

            property_concept_dict[property_id] = concept_ids

        self.data_df.reset_index(inplace=True)

        return concept_property_dict, property_concept_dict

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        return self.data_df["concept"][idx], self.data_df["property"][idx]

    def add_context(self, batch):

        if self.context_num == 1:

            concept_context = "Concept : "
            property_context = "Property : "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 2:

            concept_context = "The notion we are modelling : "
            property_context = "The notion we are modelling : "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 3:

            prefix_num = 2
            suffix_num = 2

            print("prefix_num :", prefix_num)
            print("suffix_num :", suffix_num)

            concepts_batch = [
                "[MASK] " * prefix_num + concept + " " + "[MASK] " * suffix_num + "."
                for concept in batch[0]
            ]
            property_batch = [
                "[MASK] " * prefix_num + prop + " " + "[MASK] " * suffix_num + "."
                for prop in batch[1]
            ]
        elif self.context_num == 4:

            concept_context = "Yesterday, I saw another "
            property_context = "Yesterday, I saw a thing which is "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        elif self.context_num == 5:

            concept_context = "The notion we are modelling is called "
            property_context = "The notion we are modelling is called "

            concepts_batch = [concept_context + x + "." for x in batch[0]]
            property_batch = [property_context + x + "." for x in batch[1]]

        return concepts_batch, property_batch

    def tokenize(
        self, concept_batch, property_batch, concept_max_len=64, property_max_len=64
    ):

        concept_ids = self.tokenizer(
            concept_batch,
            max_length=concept_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        property_ids = self.tokenizer(
            property_batch,
            max_length=property_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "concept_inp_id": concept_ids.get("input_ids"),
            "concept_atten_mask": concept_ids.get("attention_mask"),
            "property_inp_id": property_ids.get("input_ids"),
            "property_atten_mask": property_ids.get("attention_mask"),
        }

