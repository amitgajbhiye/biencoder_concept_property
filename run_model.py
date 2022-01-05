import argparse
import logging
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

    criterion = nn.CrossEntropyLoss()

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
            logits, probs, preds = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
            )

            log.info(f"\nlogits : {logits}")
            log.info(f"\nprobs: {probs}")
            log.info(f"\npreds : {preds}")

            loss = criterion(logits, label)
            loss.backward()  # Model backward pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_preds.append(preds)
            epoch_labels.append(label)

            if step % config["training_params"]["printout_freq"] == 0 and not step == 0:
                batch_scores = compute_scores(label, preds)

                log.info(
                    f"  Batch {step} of Batch {len(train_dataloader)} -- > \
                        loss : {loss}, Binary F1: {batch_scores.get('binary_f1')}"
                )

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        log.info(
            f"epoch_labels type: {type((epoch_labels))} {np.concatenate(epoch_labels)}"
        )
        log.info(
            f"torch.flatten(epoch_labels): {torch.flatten(torch.tensor(epoch_labels))}"
        )

        epoch_scores = compute_scores(
            torch.flatten(epoch_labels), torch.flatten(epoch_preds)
        )

        log.info(f"\nEpoch {epoch} finished !!")
        log.info(f"  Average Epoch Loss: {avg_train_loss}")
        log.info("Epoch Scores")
        log.info(epoch_scores)

        log.info(f"\n Running Validation ...")
        model.eval()

        val_loss, val_preds, val_labels = [], [], []

        with torch.no_grad:
            for batch in val_dataloader:
                (
                    concept_inp_id,
                    concept_attention_mask,
                    property_input_id,
                    property_attention_mask,
                    label,
                ) = [val.to(device) for _, val in batch.items()]

                # Model forward pass
                logits, probs, preds = model(
                    concept_input_id=concept_inp_id,
                    concept_attention_mask=concept_attention_mask,
                    property_input_id=property_input_id,
                    property_attention_mask=property_attention_mask,
                )

                loss = criterion(logits, label)

                val_loss += loss.item()
                val_preds.append(preds)
                val_labels.append(label)

        val_scores = compute_scores(torch.flatten(val_labels, torch.flatten(val_preds)))
        log.info("\nValidation Scores")
        log.info(val_scores)


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
    # display(item=config)
    log.info(config)

    train(config)
