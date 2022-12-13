import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data._utils.collate import default_convert
from model.joint_encoder_concept_property import (
    DatasetConceptProperty,
    ModelConceptProperty,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# test_file = "data/generate_embeddding_data/mcrae_related_data/dummy.txt"

test_file = "data/train_data/joint_encoder_property_conjuction_data/false_label_con_similar_50_vocab_props.txt"
batch_size = 1024

model_save_path = "trained_models/joint_encoder_gkbcnet_cnethasprop"
model_name = (
    "joint_encoder_concept_property_gkbcnet_cnethasprop_step2_pretrained_model.pt"
)
best_model_path = os.path.join(model_save_path, model_name)

test_data = DatasetConceptProperty(test_file)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, sampler=test_sampler, collate_fn=default_convert
)

print(f"Best Model Path : {best_model_path}")
print(f"Loaded Data Frame")
print(test_data.data_df)


def predict(test_dataloader):

    print(f"Predicting From Best Saved Model : {best_model_path}", flush=True)

    model = ModelConceptProperty()

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_loss, test_accuracy, test_preds, test_logits = [], [], [], []

    for step, batch in enumerate(test_dataloader):

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        attention_mask = torch.cat([x["attention_mask"] for x in batch], dim=0).to(
            device
        )
        token_type_ids = torch.cat([x["token_type_ids"] for x in batch], dim=0).to(
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
        test_loss.append(loss.item())

        batch_preds = torch.argmax(logits, dim=1).flatten()

        batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100

        test_accuracy.append(batch_accuracy)
        test_preds.extend(batch_preds.cpu().detach().numpy())

        test_logits.extend(torch.sigmoid(logits).cpu().detach().numpy())

    loss = np.mean(test_loss)
    accuracy = np.mean(test_accuracy)

    return loss, accuracy, test_preds, test_logits


loss, accuracy, predictions, logit = predict(test_dataloader)


# print("logits")
# print(logits)


## Here taking the logit of the positive class - What the model thinks about the propety.
## That is how confident the model is about the property that it applies to concepts

positive_class_logits = [round(l[1], 4) for l in logit]


print(f"Number of Logits : {len(logit)}")
print(f"test_data.data_df.shape[0] : {test_data.data_df.shape[0]}")

# print(f"Logits: {logit}")
# print(f"positive_class_logits: {positive_class_logits}")

assert test_data.data_df.shape[0] == len(
    positive_class_logits
), "length of test dataframe is not equal to logits"

new_test_dataframe = test_data.data_df.copy(deep=True)
new_test_dataframe.drop("labels", axis=1, inplace=True)
new_test_dataframe["logit"] = positive_class_logits

unique_concepts = new_test_dataframe["concept"].unique()

######### Taking Top K records  #########

# top_k_prop = 20
# all_data_list = []
# for concept in unique_concepts:

#     con_df = new_test_dataframe[new_test_dataframe["concept"] == concept].sort_values(
#         by="logit", ascending=False
#     )
#     con_df = con_df[0:top_k_prop]
#     all_data_list.extend(con_df.values.tolist())

# top_k_df_with_logit = pd.DataFrame.from_records(
#     all_data_list, columns=["concept", "property", "logit"]
# )


# logit_filename = f"data/generate_embeddding_data/mcrae_related_data/with_logits_bert_base_gkb_cnet_trained_model_mcrae_concept_top_{top_k_prop}similar_properties.tsv"
# top_k_df_with_logit.to_csv(logit_filename, sep="\t", index=None, header=None)
# print(top_k_df_with_logit.head(n=20), flush=True)


# top_k_df_with_logit.drop(labels="logit", axis=1, inplace=True)
# logit_filename = f"data/generate_embeddding_data/mcrae_related_data/bert_base_gkb_cnet_trained_model_mcrae_concept_top_{top_k_prop}similar_properties.tsv"
# top_k_df_with_logit.to_csv(logit_filename, sep="\t", index=None, header=None)

# print()
# print(top_k_df_with_logit.head(n=20), flush=True)


######### Taking records with positive class threshold of 0.50 #########

df_with_threshold_50 = new_test_dataframe[new_test_dataframe["logit"] > 0.50]

logit_filename = "data/train_data/joint_encoder_property_conjuction_data/with_logits_con_similar_50_vocab_props.txt"
df_with_threshold_50.to_csv(logit_filename, sep="\t", index=None, header=None)
print(df_with_threshold_50.head(n=20), flush=True)


df_with_threshold_50.drop(labels="logit", axis=1, inplace=True)
logit_filename = "data/train_data/joint_encoder_property_conjuction_data/je_filtered_con_similar_vocab_properties.txt"
df_with_threshold_50.to_csv(logit_filename, sep="\t", index=None, header=None)

print()
print(df_with_threshold_50.head(n=20), flush=True)


# predictions = np.array(predictions).flatten()


# print("Test Loss :", loss, flush=True)
# print("Test Accuracy :", accuracy, flush=True)

# test_glod_labels = test_data.labels.cpu().detach().numpy()

# with open(
#     "saved_model/musubu_models/musubu_70k_best_model_predictions.txt", "w"
# ) as pred_file:
#     for pred in predictions:
#         pred_file.write(f"{int(pred)}\n")

# print("\n", "#" * 50, flush=True)
# print("predictions :", predictions, flush=True)


# print("\nTest Metrices", flush=True)

# print(len(predictions))

# print(len(test_data.labels))

# print("\nAccuracy : ", round(accuracy_score(test_glod_labels, predictions) * 100), 4)

# print(
#     "\nF1 Score Binary: ",
#     round(f1_score(test_glod_labels, predictions, average="binary"), 4),
# )
# print(
#     "\nF1 Score Micro: ",
#     round(f1_score(test_glod_labels, predictions, average="micro"), 4),
# )
# print(
#     "\nF1 Score Macro: ",
#     round(f1_score(test_glod_labels, predictions, average="macro"), 4),
# )
# print(
#     "\nF1 Score Weighted: ",
#     round(f1_score(test_glod_labels, predictions, average="weighted"), 4),
# )

# print("\nClassification Report: ")
# print(classification_report(test_glod_labels, predictions, labels=[0, 1]))

# print("\nConfusion Matrix: ")
# print(confusion_matrix(test_glod_labels, predictions, labels=[0, 1]))


# from collections import Counter

# print("Counter")
# print(Counter(list(predictions)))

