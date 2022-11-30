import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data._utils.collate import default_convert
from joint_encoder import MusubuModel, ConPropDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# test_file = "data/generate_embeddding_data/mcrae_related_data/with_false_label_bert_base_gkb_cnet_trained_model_mcrae_concept_similar_properties.tsv"
test_file = "data/generate_embeddding_data/mcrae_related_data/dummy.txt"
batch_size = 256

model_save_path = "trained_models/joint_encoder_gkbcnet_cnethasprop"
model_name = "joint_encoder_gkbcnet_cnethasprop.pt"
best_model_path = os.path.join(model_save_path, model_name)

test_data = ConPropDataset(test_file)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, sampler=test_sampler, collate_fn=default_convert
)

print(f"Loaded Data Frame")
print(test_data.data_df)


def predict(test_dataloader):

    print(f"Predicting From Best Saved Model : {best_model_path}", flush=True)

    model = MusubuModel()

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_loss, test_accuracy, test_preds, test_logits = [], [], [], []

    for step, batch in enumerate(test_dataloader):

        input_ids = torch.cat([x["input_ids"] for x in batch], dim=0).to(device)
        attention_masks = torch.cat([x["attention_masks"] for x in batch], dim=0).to(
            device
        )
        labels = torch.tensor([x["labels"] for x in batch]).to(device)

        with torch.no_grad():
            loss, logits = model(input_ids, attention_masks, labels)

        test_loss.append(loss.item())

        batch_preds = torch.argmax(logits, dim=1).flatten()

        batch_accuracy = (labels == batch_preds).cpu().numpy().mean() * 100

        test_accuracy.append(batch_accuracy)
        test_preds.extend(batch_preds.cpu().detach().numpy())

        test_logits.extend(logits.cpu().detach().numpy())

    loss = np.mean(test_loss)
    accuracy = np.mean(test_accuracy)

    return loss, accuracy, test_preds, test_logits


loss, accuracy, predictions, logits = predict(test_dataloader)

assert test_data.data_df.shape == len(
    logits
), "length of test dataframe is not equal to logits"

new_test_dataframe = test_data.data_df.copy(deep=True)

new_test_dataframe.drop("labels", axis=1, inplace=True)

new_test_dataframe["logits"] = logits


logit_filename = "data/generate_embeddding_data/mcrae_related_data/with_logits_bert_base_gkb_cnet_trained_model_mcrae_concept_similar_properties.tsv"
new_test_dataframe.to_csv(logit_filename, sep="\t", index=None, header=None)

print(new_test_dataframe.head(n=20), flush=True)


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

