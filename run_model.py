import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.std import trange

from utils.functions import (
    compute_scores,
    create_dataloader,
    create_model,
    display,
    read_config,
)
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

log = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(config):

    log.info("Initialising datasets...")
    train_dataloader = create_dataloader(config.get("train_dataset_params"))
    val_dataloader = create_dataloader(config.get("val_dataset_params"))

    log.info("Initialising Model...")

    model = create_model(config.get("model_params"))
    model.to(device)

    # -------------------- Preparation for training  ------------------- #

    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=config["training_params"].get("lr"))

    total_training_steps = (
        len(train_dataloader) * config["training_params"]["max_epochs"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training_params"]["num_warmup_steps"],
        num_training_steps=total_training_steps,
    )

    best_val_f1 = 0.0
    start_epoch = 1

    epoch_count = []
    train_losses = []
    val_losses = []

    log.info(f"\nTraining the concept property model on {device}")

    patience_counter = 0

    for epoch in trange(start_epoch, config["training_params"].get("max_epochs")):

        epoch_count.append(epoch)
        epoch_loss = 0.0
        epoch_preds, epoch_labels = [], []

        log.info(f"  Epoch {epoch} of {config['training_params'].get('max_epochs')}")
        print("\n", flush=True)

        model.train()

        for step, batch in enumerate(train_dataloader):
            model.zero_grad()

            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
                label,
            ) = [val.to(device) for _, val in batch.items()]

            # Model forward pass
            logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
            )

            log.info(f"\nlogits shape: {logits.shape}")
            log.info(f"logits: {logits}")
            log.info(f"\nlabel shape: {label.shape}")
            log.info(f"label: {label}")

            loss = criterion(logits, label)
            loss.backward()  # Model backward pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_preds.append(preds.detach().cpu().numpy())
            epoch_labels.append(label.detach().cpu().numpy())

            if step % config["training_params"]["printout_freq"] == 0 and not step == 0:
                batch_scores = compute_scores(
                    label.detach().cpu().numpy(), preds.detach().cpu().numpy()
                )

                log.info(
                    f"  Batch {step} of Batch {len(train_dataloader)} -- > loss : {loss}, Binary F1: {batch_scores.get('binary_f1')}"
                )
                print(flush=True)

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        log.info(
            f"epoch_labels flattened: {type(np.concatenate(epoch_labels))} {np.concatenate(epoch_labels)}"
        )

        log.info(
            f"epoch_preds flattened: {type(np.concatenate(epoch_preds))} {np.concatenate(epoch_preds)}"
        )

        epoch_scores = compute_scores(
            np.concatenate(epoch_labels), np.concatenate(epoch_preds)
        )

        log.info(f"\nEpoch {epoch} finished !!")
        log.info(f"  Average Epoch Loss: {avg_train_loss}")
        log.info("Train Scores")
        for key, value in epoch_scores.items():
            log.info(f"{key} : {value}")

        # ---------------Validation---------------------#
        log.info(f"\n Running Validation ...")
        print(flush=True)
        model.eval()

        val_loss, best_val_f1, val_preds, val_labels = 0.0, 0.0, [], []

        for batch in val_dataloader:
            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
                label,
            ) = [val.to(device) for _, val in batch.items()]

            with torch.no_grad():  # Model forward pass
                logits = model(
                    concept_input_id=concept_inp_id,
                    concept_attention_mask=concept_attention_mask,
                    property_input_id=property_input_id,
                    property_attention_mask=property_attention_mask,
                )

            loss = criterion(logits, label)

            preds = torch.round(torch.sigmoid(logits))
            val_loss += loss.item()
            val_preds.append(preds.detach().cpu().numpy())
            val_labels.append(label.detach().cpu().numpy())

        val_scores = compute_scores(
            np.concatenate(val_labels), np.concatenate(val_preds)
        )

        val_binary_f1 = val_scores.get("binary_f1")

        if val_binary_f1 <= best_val_f1:
            patience_counter += 1
        else:
            patience_counter = 0
            best_val_f1 = val_binary_f1
            torch.save(
                model.state_dict(),
                os.path.join(
                    config["training_params"]["export_path"], "biencoder_bert_best.pt",
                ),
            )
        log.info("\nValidation Scores")
        log.info(f" Best Validation F1 yet : {best_val_f1}")

        for key, value in val_scores.items():
            log.info(f"{key} : {value}")

        print(flush=True)

        if patience_counter >= config["training_params"].get("early_stopping_patience"):
            break

        print(flush=True)


def evaluate(config):

    log.info(f"\n {'*' * 50}")
    log.info(f"Testing the best model")

    model = create_model(config.get("model_params"))

    best_model_path = os.path.join(
        config["training_params"]["export_path"], "biencoder_bert_best.pt",
    )

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_dataloader = create_dataloader(config.get("test_dataset_params"))

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
                label,
            ) = [val.to(device) for _, val in batch.items()]

            logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
            )
            preds = torch.round(torch.sigmoid(logits))
            test_preds.append(preds.detach().cpu().numpy())
            test_labels.append(label.detach().cpu().numpy())

    all_labels = np.concatenate(test_labels)
    all_preds = np.concatenate(test_preds)

    test_scores = compute_scores(all_labels, all_preds)

    log.info(f"Test Metrices")
    for key, value in test_scores.items():
        log.info(f"{key} : {value}")
    print(flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Siamese Concept Property Model")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The model is run with the following configuration")

    log.info(f"\n {config} \n")

    train(config)

    evaluate(config)
