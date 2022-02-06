import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TestDataset(Dataset):
    def __init__(self, dataset_params):

        self.data_df = pd.read_csv(
            dataset_params.get("test_file_path"),
            sep="\t",
            header=None,
            names=["concept", "property", "label"],
        )

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(
            dataset_params.get("hf_tokenizer_path")
        )

        self.context_num = dataset_params.get("context_num")

        self.label = self.data_df["label"].values

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

