import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle

from argparse import ArgumentParser

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from utils.functions import compute_scores

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"The Model is Trained on : {device}")

#### Parameters ####

# MODEL_CLASS = {
#     "bert-base-seq-classification": (
#         BertForSequenceClassification,
#         "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer",
#         "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model",
#     ),
# }

print(f"Property Conjuction Joint Encoder Model- Step3")

hawk_bb_tokenizer = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer"
hawk_bb_model = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model"

data_path = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data"


# train_file = os.path.join(
#     data_path, "5_neg_train_random_and_similar_conjuct_properties.tsv"
# )

# valid_file = os.path.join(
#     data_path, "5_neg_valid_random_and_similar_conjuct_properties.tsv"
# )

train_file = None
valid_file = None
test_file = None

# train_file = os.path.join(data_path, "dummy_prop_conj.tsv")
# valid_file = os.path.join(data_path, "dummy_prop_conj.tsv")

print(f"Train File : {train_file}")
print(f"Valid File : {valid_file}")

model_save_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop"
model_name = "joint_encoder_property_conjuction_random_similar_props_gkbcnet_cnethasprop_step3_pretrained_model.pt"
best_model_path = os.path.join(model_save_path, model_name)

max_len = 200

num_labels = 2
batch_size = 64
# num_epoch = 100
num_epoch = 12
lr = 2e-6


pretrained_model_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop/joint_encoder_property_conjuction_random_similar_props_gkbcnet_cnethasprop_step3_pretrained_model.pt"
# cv_type = "concept_split"
# num_fold = 5
load_pretrained = True


class DatasetPropConjuction(Dataset):
    def __init__(self, concept_property_file, max_len=max_len):

        if isinstance(concept_property_file, pd.DataFrame):

            self.data_df = concept_property_file
            print(
                f"Supplied Concept Property File is a Dataframe : {self.data_df.shape}"
            )

        else:

            print(f"Supplied Concept Property File is a Path : {concept_property_file}")
            print(f"Loading into Dataframe ... ")

            self.data_df = pd.read_csv(
                concept_property_file,
                sep="\t",
                header=None,
                names=["concept", "conjuct_prop", "predict_prop", "labels"],
            )

            print(f"Loaded Dataframe Shape: {self.data_df.shape}")

        self.tokenizer = BertTokenizer.from_pretrained(hawk_bb_tokenizer)
        self.max_len = max_len

        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

        self.labels = torch.tensor(self.data_df["labels"].values)

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        concept = self.data_df["concept"][idx].replace(".", "").strip()
        conjuct_props = self.data_df["conjuct_prop"][idx].strip()
        predict_prop = self.data_df["predict_prop"][idx].replace(".", "").strip()
        labels = self.data_df["labels"][idx]

        if conjuct_props == "Nothing to Conjuct":

            con_prop_conj = concept + " " + self.sep_token
            prop_to_predict = predict_prop + " "

        else:

            con_prop_conj = concept + " " + self.sep_token + " " + conjuct_props
            prop_to_predict = predict_prop + " "

        # print(f"{con_prop_conj} - {prop_to_predict} - {labels.item()}", flush=True)

        encoded_dict = self.tokenizer.encode_plus(
            text=con_prop_conj,
            text_pair=prop_to_predict,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        input_ids = encoded_dict["input_ids"]
        attention_mask = encoded_dict["attention_mask"]
        token_type_ids = encoded_dict["token_type_ids"]

        # print(f"input_ids : {input_ids}")
        # print(f"attention_mask : {attention_mask}")
        # print(f"token_type_ids : {token_type_ids}")
        # print(f"labels :", {labels})
        # print()

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ModelPropConjuctionJoint(nn.Module):
    def __init__(self):
        super(ModelPropConjuctionJoint, self).__init__()

        # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(
            hawk_bb_model, num_labels=num_labels
        )

        assert self.bert.config.num_labels == 2

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss, logits = output.loss, output.logits

        return loss, logits


def load_pretrained_model(pretrained_model_path):

    model = ModelPropConjuctionJoint()
    model.load_state_dict(torch.load(pretrained_model_path))

    print(f"The pretrained Model is loaded from : {pretrained_model_path}")

    return model


def prepare_data_and_models(train_file, valid_file, test_file, load_pretrained):

    train_data = DatasetPropConjuction(train_file)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=default_convert,
    )

    if valid_file is not None:
        val_data = DatasetPropConjuction(valid_file)
        val_sampler = RandomSampler(val_data)
        val_dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=default_convert,
        )
    else:
        val_dataloader = None

    if test_file is not None:
        test_data = DatasetPropConjuction(test_file)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=default_convert,
        )
    else:
        test_dataloader = None

    if load_pretrained:
        model = load_pretrained_model(pretrained_model_path)
    else:
        model = ModelPropConjuctionJoint()

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    return (
        model,
        scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


def train_on_single_epoch(model, scheduler, optimizer, train_dataloader):

    total_epoch_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        loss, logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and not step == 0:
            print(
                "   Batch {} of Batch {} ---> Batch Loss {}".format(
                    step, len(train_dataloader), round(loss.item(), 4)
                ),
                flush=True,
            )

    avg_train_loss = total_epoch_loss / len(train_dataloader)

    print("Average Train Loss :", round(avg_train_loss, 4), flush=True)

    return avg_train_loss, model


def evaluate(model, dataloader):

    model.eval()

    val_loss, val_accuracy, val_preds, val_labels = [], [], [], []

    for step, batch in enumerate(dataloader):

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
            device
        )
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        val_loss.append(loss.item())

        batch_preds = torch.argmax(logits, dim=1).flatten()
        batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100

        val_accuracy.append(batch_accuracy)
        val_preds.extend(batch_preds.cpu().detach().numpy())
        val_labels.extend(labels.cpu().detach().numpy())

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()

    return val_loss, val_preds, val_labels


def train(
    model,
    scheduler,
    optimizer,
    train_dataloader,
    val_dataloader=None,
    test_dataloader=None,
):

    best_valid_loss, best_valid_f1 = 0.0, 0.0

    patience_early_stopping = 10
    patience_counter = 0
    start_epoch = 1

    train_losses, valid_losses = [], []

    for epoch in range(start_epoch, num_epoch + 1):

        print("\n Epoch {:} of {:}".format(epoch, num_epoch), flush=True)

        train_loss, model = train_on_single_epoch(
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
        )

        if val_dataloader is not None:

            print(f"Running Validation ....")

            valid_loss, valid_preds, valid_gold_labels = evaluate(
                model=model, dataloader=val_dataloader
            )

            scores = compute_scores(valid_gold_labels, valid_preds)
            valid_binary_f1 = scores["binary_f1"]

            if valid_binary_f1 < best_valid_f1:
                patience_counter += 1
            else:
                patience_counter = 0
                best_valid_f1 = valid_binary_f1

                print("\n", "+" * 20)
                print("Saving Best Model at Epoch :", epoch, model_name)
                print("Epoch :", epoch)
                print("   Best Validation F1:", best_valid_f1)

                torch.save(model.state_dict(), best_model_path)

                print(f"The best model is saved at : {best_model_path}")

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print("\n", flush=True)
            print("+" * 50, flush=True)

            print("valid_preds shape:", valid_preds.shape)
            print("val_gold_labels shape:", valid_gold_labels.shape)

            print(f"\nTraining Loss: {round(train_loss, 4)}", flush=True)
            print(f"Validation Loss: {round(valid_loss, 4)}", flush=True)

            print(f"Current Validation F1 Score Binary {valid_binary_f1}", flush=True)
            print(f"Best Validation F1 Score Yet : {best_valid_f1}", flush=True)

            print("Validation Scores")
            for key, value in scores.items():
                print(f" {key} :  {value}")

            if patience_counter > patience_early_stopping:

                print(f"\nTrain Losses :", train_losses, flush=True)
                print(f"Validation Losses: ", valid_losses, flush=True)

                print("Early Stopping ---> Patience Reached!!!")
                break

        elif test_dataloader is not None:
            print(f"Testing the Model ....")

            _, test_preds, test_gold_labels = evaluate(model, val_dataloader)

            print("************ Test Scores ************")
            for key, value in scores.items():
                print(f" {key} :  {value}")

            return test_preds, test_gold_labels
        else:

            print(f"Valid Dataloader :{val_dataloader}")
            print(f"Test Dataloader :{test_dataloader}")

            print(f"Valid and Test Dataloaders both cant be None")


################ Fine Tuning Code Starts Here ################

# ft_param_dict = {"pretrained_model_path": "", "cv_type": "concept_split", "num_fold": 5}


def concept_split_training(train_file, test_file, load_pretrained):

    print(f"Training the Model on Concept Split")
    print(f"Train File : {train_file}")
    print(f"Test File : {test_file}")
    print(f"Load Pretrained :{load_pretrained}")

    (
        model,
        scheduler,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = prepare_data_and_models(
        train_file=train_file,
        valid_file=None,
        test_file=test_file,
        load_pretrained=load_pretrained,
    )

    train(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )


def do_cv(cv_type):

    if cv_type == "concept_split":
        pass
        # concept_split_training(train_file, test_file, load_pretrained)

    elif cv_type in ("property_split", "concept_property_split"):

        if cv_type == "property_split":

            num_fold = 5
            dir_name = "/scratch/c.scmag3/biencoder_concept_property/data/evaluation_data/mcrae_joint_encoder_prop_conjuction_fine_tune/property_split"
            train_file_base_name = "prop_conj_train_prop_split_con_prop.pkl"
            test_file_base_name = "prop_conj_test_prop_split_con_prop.pkl"

            print(f"Training the Property Split")
            print(f"Number of Folds: {num_fold}")

        elif cv_type == "concept_property_split":

            num_fold = 9
            dir_name = "data/evaluation_data/mcrae_con_prop_split_train_test_files"
            train_file_base_name = "train_con_prop_split_con_prop.pkl"
            test_file_base_name = "test_con_prop_split_con_prop.pkl"

            print(f"Training the Concept Property Split")
            print(f"Number of Folds: {num_fold}")

        else:
            raise Exception(f"Specify a correct Split")

        all_folds_test_preds, all_folds_test_labels = [], []
        for fold in range(num_fold):

            print(f"Training the model on Fold : {fold}/{num_fold}")

            train_file_name = os.path.join(dir_name, f"{fold}_{train_file_base_name}")
            test_file_name = os.path.join(dir_name, f"{fold}_{test_file_base_name}")

            print(f"Train File Name : {train_file_name}")
            print(f"Test File Name : {test_file_name}")

            # with open(train_file_name, "rb") as train_pkl, open(
            #     test_file_name, "rb"
            # ) as test_pkl:

            #     train_df = pickle.load(train_pkl)
            #     test_df = pickle.load(train_pkl)

            (
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = prepare_data_and_models(
                train_file=train_file_name,
                valid_file=None,
                test_file=test_file_name,
                load_pretrained=load_pretrained,
            )

            fold_test_preds, fold_test_gold_labels = train(
                model,
                scheduler,
                optimizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
            )

            all_folds_test_preds.extend(fold_test_preds)
            all_folds_test_labels.extend(fold_test_gold_labels)

            scores = compute_scores(fold_test_gold_labels, fold_test_preds)

            print(f"Scores for Fold : {fold} ")

            for key, value in scores.items():
                print(f"{key} : {value}")

        all_folds_test_preds = np.array(all_folds_test_preds).flatten()
        all_folds_test_labels = np.array(all_folds_test_labels).flatten()

        print(f"Shape of All Folds Preds : {all_folds_test_preds.shape}")
        print(f"Shape of All Folds Labels : {all_folds_test_labels.shape}")

        assert (
            all_folds_test_preds.shape == all_folds_test_labels.shape
        ), "shape of all folds labels not equal to all folds preds"

        print(f"Calculating the scores for All Folds")

        scores = compute_scores(all_folds_test_labels, all_folds_test_preds)

        for key, value in scores.items():
            print(f"{key} : {value}")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Joint Encoder Property Conjuction Model - Step 3"
    )

    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--cv_type",)

    args = parser.parse_args()

    print(f"Supplied Arguments")
    print("args.pretrain :", args.pretrain)
    print("args.finetune:", args.finetune)

    if args.pretrain:

        (
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = prepare_data_and_models(
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
            load_pretrained=load_pretrained,
        )

        train(
            model,
            scheduler,
            optimizer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        )

    elif args.finetune:

        print(f"Finetuning the Pretrained Model")

        cv_type = args.cv_type

        do_cv(cv_type=cv_type)

