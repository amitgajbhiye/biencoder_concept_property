from os import sep
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ConceptPropertyDataset(Dataset):
    def __init__(self, dataset_params):

        self.data_df = pd.read_csv(
            dataset_params.get("dataset_path"),
            sep="\t",
            header=None,
            names=["concept", "property", "label"],
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            dataset_params.get("hf_tokenizer_path")
        )

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.concept_max_length = dataset_params.get("concept_max_len", 15)
        self.property_max_length = dataset_params.get("property_max_len", 20)

        self.concept_encodings = self.generate_inp_ids(
            self.data_df["concept"].tolist(), max_length=self.concept_max_length
        )
        self.property_encodings = self.generate_inp_ids(
            self.data_df["property"].tolist(), max_length=self.property_max_length
        )

    def generate_inp_ids(self, text_list, max_length):

        encodings = self.tokenizer(
            text_list,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return encodings

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        concept_data = {
            key: torch.tensor(val[idx]) for key, val in self.concept_encodings.items()
        }
        property_data = {
            key: torch.tensor(val[idx]) for key, val in self.property_encodings.items()
        }
        label = torch.tensor(self.data_df.iloc[idx, 2])

        return {
            "concept_inp_id": concept_data.get("input_ids"),
            "concept_atten_mask": concept_data.get("attention_mask"),
            "property_inp_id": property_data.get("input_ids"),
            "property_atten_mask": property_data.get("attention_mask"),
            "label": label,
        }

