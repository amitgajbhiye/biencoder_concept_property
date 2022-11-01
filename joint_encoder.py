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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

import warnings

warnings.filterwarnings("ignore")


# In[ ]:


print(
    "Fintuning the BERT Large Uncased model in Musubu Model with 'MSCG + Prefix_Adjectives + GKB_Properties' Data"
)


# In[ ]:


# tokenizer_bert_large_uncased_path = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_large_uncased/tokenizer"
# model_bert_large_uncased_path = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_large_uncased/model"


bb_tokenizer = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/tokenizer"
bb_model = "/scratch/c.scmag3/conceptEmbeddingModel/for_seq_classification_bert_base_uncased/model"


# In[ ]:


class ConPropDataset(Dataset):
    def __init__(self, concept_property_file, max_len=128):

        self.data_df = pd.read_csv(
            concept_property_file,
            sep="\t",
            header=None,
            names=["concept", "property", "labels"],
        )

        self.tokenizer = BertTokenizer.from_pretrained(bb_tokenizer)

        self.max_len = max_len
        self.sep_token = self.tokenizer.sep_token

        self.labels = torch.tensor(self.data_df["labels"].values)

    def __len__(self):

        return len(self.data_df)

    def __getitem__(self, idx):

        concept = self.data_df["concept"][idx]
        prop = self.data_df["property"][idx]
        labels = self.labels[idx]

        # sent = prop.replace(".", "") + " such as " + concept
        sent = concept + " " + self.sep_token + " " + prop

        print("Processing data: ", sent, flush=True)

        encoded_dict = self.tokenizer.encode_plus(
            sent,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
        }


# In[ ]:


class MusubuModel(nn.Module):
    def __init__(self):
        super(MusubuModel, self).__init__()

        # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.bert = BertForSequenceClassification.from_pretrained(
            bb_model, num_labels=2
        )

        print()
        print("self.bert.config.num_labels")
        print(self.bert.config.num_labels)

        assert self.bert.config.num_labels == 2

    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids, attention_mask=attention_mask, labels=labels)

        loss, logits = output.loss, output.logits

        return loss, logits


# In[ ]:


batch_size = 32
num_epoch = 100


train_file = (
    "data/train_data/joint_encoder/5_neg_train_gkbcnet_plus_cnethasproperty.tsv"
)
valid_file = "data/train_data/joint_encoder/5_neg_val_gkbcnet_plus_cnethasproperty.tsv"
# test_file = "/scratch/c.scmag3/conceptEmbeddingModel/data/70k_test_ms_concept_graph.tsv"


train_data = ConPropDataset(train_file)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,)

val_data = ConPropDataset(valid_file)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)


# In[ ]:


model = MusubuModel()
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-6)

total_steps = len(train_dataloader) * num_epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


# In[ ]:


def train():

    total_epoch_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        input_ids = torch.tensor([x["input_ids"] for x in batch]).to(device)
        attention_masks = torch.tensor([x["attention_masks"] for x in batch]).to(device)
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        # input_ids = batch["input_ids"].to(device)
        # attention_masks = batch["attention_masks"].to(device)
        # labels = torch.tensor([x["labels"] for x in batch]).to(device)

        loss, logits = model(input_ids, attention_masks, labels)

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


# In[ ]:


def evaluate():

    print("\n Running Validation...")

    model.eval()

    val_loss, val_accuracy, valid_preds = [], [], []

    for step, batch in enumerate(val_dataloader):

        input_ids = torch.tensor([x["input_ids"] for x in batch], dim=0).to(device)
        attention_masks = torch.tensor([x["attention_masks"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        with torch.no_grad():
            loss, logits = model(input_ids, attention_masks, labels)

        val_loss.append(loss.item())

        batch_preds = torch.argmax(logits, dim=1).flatten()

        batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100

        val_accuracy.append(batch_accuracy)
        valid_preds.extend(batch_preds.cpu().detach().numpy())

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    valid_preds = np.array(valid_preds).flatten()

    return val_loss, val_accuracy, valid_preds


# In[ ]:


best_valid_loss = 0.0
best_valid_f1 = 0.0

patience_early_stopping = 10
patience_counter = 0
start_epoch = 1

model_save_path = "----"
model_name = "----"

best_model_path = os.path.join(model_save_path, model_name)


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


# In[ ]:


# In[ ]:


#

# test_data = ConPropDataset(test_file)
#
# test_sampler = SequentialSampler(test_data)
# test_dataloader = DataLoader(test_data,
#                              batch_size = batch_size,
#                              sampler = test_sampler,
#                              collate_fn=default_convert)

# def predict(test_dataloader):
#
#     print("\n Predicting From Best Saved Model...", flush=True)
#
#     best_model_path = os.path.join('saved_model/musubu_models', model_name)
#
#     model = MusubuModel()
#
#     model.load_state_dict(torch.load(best_model_path))
#     model.eval()
#     model.to(device)
#
#
#     test_loss, test_accuracy, test_preds = [], [], []
#
#     for step, batch in enumerate(test_dataloader):
#
#         input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
#         attention_masks = torch.cat([x["attention_masks"] for x in batch], dim=0).to(device)
#         labels = torch.tensor([x["labels"] for x in batch]).to(device)
#
#         with torch.no_grad():
#             loss, logits = model(input_ids, attention_masks, labels)
#
#         test_loss.append(loss.item())
#
#         batch_preds = torch.argmax(logits, dim=1).flatten()
#
#         batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100
#
#         test_accuracy.append(batch_accuracy)
#         test_preds.extend(batch_preds.cpu().detach().numpy())
#
#     loss = np.mean(test_loss)
#     accuracy = np.mean(test_accuracy)
#
#     return loss, accuracy, test_preds
#

# loss, accuracy, predictions = predict(test_dataloader)
#
# predictions = np.array(predictions).flatten()
#
# print ('Test Loss :', loss, flush=True)
# print ('Test Accuracy :', accuracy, flush=True)
#
# test_glod_labels = test_data.labels.cpu().detach().numpy()
#
# with open ('saved_model/musubu_models/musubu_70k_best_model_predictions.txt', 'w') as pred_file:
#     for pred in predictions:
#         pred_file.write(f'{int(pred)}\n')
#
# print ('\n', '#' * 50, flush=True)
# print ('predictions :', predictions, flush=True)
#

# print ('\nTest Metrices', flush=True)

# print (len(predictions))

# print (len(test_data.labels))

# print ('\nAccuracy : ', round(accuracy_score(test_glod_labels, predictions) * 100), 4)

# print ('\nF1 Score Binary: ', round(f1_score(test_glod_labels, predictions, average='binary'), 4))
# print ('\nF1 Score Micro: ', round(f1_score(test_glod_labels, predictions, average='micro'), 4))
# print ('\nF1 Score Macro: ', round(f1_score(test_glod_labels, predictions, average='macro'), 4))
# print ('\nF1 Score Weighted: ', round(f1_score(test_glod_labels, predictions, average='weighted'), 4))

# print ('\nClassification Report: ')
# print (classification_report(test_glod_labels, predictions, labels=[0, 1]))

# print ('\nConfusion Matrix: ')
# print (confusion_matrix(test_glod_labels, predictions, labels=[0, 1]))

#
# from collections import Counter
# print ('Counter')
# print(Counter(list(predictions)))

#

