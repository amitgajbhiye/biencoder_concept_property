import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

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


hawk_bb_tokenizer = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer"
hawk_bb_model = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model"

data_path = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data"

train_file = os.path.join(
    data_path, "5_neg_train_random_and_similar_conjuct_properties.tsv"
)
valid_file = os.path.join(
    data_path, "5_neg_valid_random_and_similar_conjuct_properties.tsv"
)

model_save_path = "/scratch/c.scmag3/biencoder_concept_property/trained_models/joint_encoder_gkbcnet_cnethasprop"
model_name = (
    "joint_encoder_random_similar_prop_conj_gkbcnet_cnethasprop_pretrained_model.pt"
)
best_model_path = os.path.join(model_save_path, model_name)

max_len = 128

num_labels = 2

patience_early_stopping = 10
patience_counter = 0
start_epoch = 1
batch_size = 32
num_epoch = 100
lr = 2e-6


class DatasetPropConjuction(Dataset):
    def __init__(self, concept_property_file, max_len=max_len):

        self.data_df = pd.read_csv(
            concept_property_file,
            sep="\t",
            header=None,
            names=["concept", "conjuct_prop", "predict_prop", "labels"],
        )

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
        labels = self.labels[idx]

        if conjuct_props == "Nothing to Conjuct":

            con_prop_conj = concept + " " + self.sep_token
            prop_to_predict = predict_prop + " "

        else:

            con_prop_conj = concept + " " + self.sep_token + " " + conjuct_props
            prop_to_predict = predict_prop + " "

        print(f"{con_prop_conj} - {prop_to_predict} - {labels.item()}")

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


train_data = DatasetPropConjuction(train_file)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=default_convert
)

val_data = DatasetPropConjuction(valid_file)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(
    val_data, batch_size=batch_size, sampler=val_sampler, collate_fn=default_convert
)

model = ModelPropConjuctionJoint()
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)

total_steps = len(train_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def train():

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
                    step, len(train_dataloader), loss
                ),
                flush=True,
            )

    avg_train_loss = total_epoch_loss / len(train_dataloader)

    print("avg_train_loss :", avg_train_loss, flush=True)

    return avg_train_loss


def evaluate():

    print("\n Running Validation...")

    model.eval()

    val_loss, val_accuracy, valid_preds = [], [], []

    for step, batch in enumerate(val_dataloader):

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
        valid_preds.extend(batch_preds.cpu().detach().numpy())

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    valid_preds = np.array(valid_preds).flatten()

    return val_loss, val_accuracy, valid_preds


best_valid_loss = 0.0
best_valid_f1 = 0.0

train_losses = []
valid_losses = []

for epoch in range(start_epoch, num_epoch + 1):

    print("\n Epoch {:} of {:}".format(epoch, num_epoch), flush=True)

    train_loss = train()
    valid_loss, valid_accuracy, valid_preds = evaluate()

    val_gold_labels = val_data.labels.cpu().detach().numpy().flatten()

    valid_binary_f1 = round(f1_score(val_gold_labels, valid_preds, average="binary"), 4)
    valid_micro_f1 = round(f1_score(val_gold_labels, valid_preds, average="micro"), 4)
    valid_macro_f1 = round(f1_score(val_gold_labels, valid_preds, average="macro"), 4)
    valid_weighted_f1 = round(
        f1_score(val_gold_labels, valid_preds, average="weighted"), 4
    )

    if valid_binary_f1 < best_valid_f1:
        patience_counter += 1
    else:
        best_valid_f1 = valid_binary_f1

        print("\n", "+" * 20)
        print("Saving Best Model at Epoch :", epoch, model_name)
        print("Epoch :", epoch)
        print("   Best Validation F1:", best_valid_f1)

        torch.save(model.state_dict(), best_model_path)

        print(f"The best model is saved at : {best_model_path}")

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print("valid_preds shape:", valid_preds.shape)
    print("val_gold_labels shape:", val_gold_labels.shape)

    print("\n", flush=True)
    print("+" * 50, flush=True)
    print(f"\nTraining Loss: {train_loss}", flush=True)
    print(f"Validation Loss: {valid_loss}", flush=True)

    print(f"Validation Accuracy: {valid_accuracy}", flush=True)
    print(
        f"sk_learn Validation Accuracy: {accuracy_score(val_gold_labels, valid_preds)}",
        flush=True,
    )

    print(f"Best Validation F1 Score Yet : {best_valid_f1}", flush=True)
    print(f"Validation F1 Score Binary {valid_binary_f1}", flush=True)
    print(f"Validation F1 Score Micro {valid_micro_f1}", flush=True)
    print(f"Validation F1 Score Macro {valid_macro_f1}", flush=True)
    print(f"Validation F1 Score Weighted {valid_weighted_f1}", flush=True)

    print(f"\nValidation Classification Report :", flush=True)
    print(
        classification_report(val_gold_labels, valid_preds, labels=[0, 1]), flush=True
    )

    print(f"\nValidation Confusion Matrix :", flush=True)
    print(confusion_matrix(val_gold_labels, valid_preds, labels=[0, 1]), flush=True)

    if patience_counter > patience_early_stopping:
        print("Early Stopping ---> Patience Reached!!!")

print(f"\nTrain Losses :", train_losses, flush=True)
print(f"Validation Losses: ", valid_losses, flush=True)
