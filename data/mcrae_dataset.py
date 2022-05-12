import logging
from os import sep

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data.concept_property_dataset import TOKENIZER_CLASS


log = logging.getLogger(__name__)


class McRaeConceptPropertyDataset(Dataset):
    def __init__(self, dataset_params, dataset_type, data_df=None):

        if dataset_type in ("train", "valid"):

            self.data_df = data_df
            self.data_df.drop_duplicates(inplace=True)
            self.data_df.dropna(inplace=True)

        elif dataset_type in ("test",):
            if data_df is not None:

                log.info(f"Loading the data from supplied DF")
                self.data_df = data_df
            else:

                log.info(
                    f"*** Loading the Test Data from 'test_file_path', DF supplied is None ***"
                )
                self.data_df = pd.read_csv(
                    dataset_params.get("test_file_path"),
                    sep="\t",
                    header=None,
                    names=["concept", "property", "label"],
                )

                self.data_df.drop_duplicates(inplace=True)
                self.data_df.dropna(inplace=True)
                self.data_df.reset_index(drop=True, inplace=True)

            log.info(f"Test Data size {self.data_df.shape}")

        ###############33#####
        self.data_df = self.data_df[0:500]

        self.hf_tokenizer_name = dataset_params.get("hf_tokenizer_name")

        self.tokenizer_class = TOKENIZER_CLASS.get(self.hf_tokenizer_name)

        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = self.tokenizer_class.from_pretrained(
            dataset_params.get("hf_tokenizer_path")
        )

        self.mask_token = self.tokenizer.mask_token

        self.context_num = dataset_params.get("context_num")

        self.label = self.data_df["label"].values

        log.info(f"hf_tokenizer_name : {dataset_params.get('hf_tokenizer_name')}")
        log.info(f"self.tokenizer_class : {self.tokenizer_class}")
        log.info(f"Mask Token for the Model : {self.mask_token}")
        log.info(f"Context Num : {self.context_num}")

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        return [
            self.data_df["concept"][idx],
            self.data_df["property"][idx],
            self.data_df["label"][idx],
        ]

    def add_context(self, batch):

        joint_con_prop_batch = []

        if self.context_num == 1:

            for con, prop in zip(batch[0], batch[1]):
                joint_con_prop_batch.append(
                    "concept " + con.strip() + " has property " + prop.strip()
                )
        elif self.context_num == 2:

            for con, prop in zip(batch[0], batch[1]):
                joint_con_prop_batch.append(
                    con.strip() + " " + self.tokenizer.sep_token + " " + prop.strip()
                )

        log.info(f"After adding context batch : {joint_con_prop_batch}")

        return joint_con_prop_batch

        # if self.context_num == 1:

        #     concept_context = "Concept : "
        #     property_context = "Property : "

        #     concepts_batch = [concept_context + x + "." for x in batch[0]]
        #     property_batch = [property_context + x + "." for x in batch[1]]

        # return concepts_batch, property_batch

    def tokenize(self, joint_con_prop_batch, max_len=64):

        if self.context_num in (1, 2, 3, 4, 5, 6):

            # # Printing for debugging
            print(f"Context Num : {self.context_num}")

            print(f"Tokenized Sentences")
            x = [self.tokenizer.tokenize(sent) for sent in joint_con_prop_batch]
            # x = self.tokenizer.tokenize(joint_con_prop_batch)
            print(x)
            y = [self.tokenizer.convert_tokens_to_ids(j) for j in x]
            print(y)

            ids = self.tokenizer(
                joint_con_prop_batch,
                add_special_tokens=True,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            print(ids)
            print()

        if self.hf_tokenizer_name in ("roberta-base", "roberta-large"):

            return {
                "inp_id": ids.get("input_ids"),
                "atten_mask": ids.get("attention_mask"),
            }
        else:

            return {
                "inp_id": ids.get("input_ids"),
                "atten_mask": ids.get("attention_mask"),
                "token_type_id": ids.get("token_type_ids"),
            }

