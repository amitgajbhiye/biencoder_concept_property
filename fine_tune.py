import logging
import os
from argparse import ArgumentParser
from cProfile import label

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.pyplot import axis
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.functions import (
    compute_scores,
    count_parameters,
    create_model,
    load_pretrained_model,
    mcrae_dataset_and_dataloader,
    read_config,
    read_data,
    set_seed,
)

log = logging.getLogger(__name__)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(
    model, train_dataset, train_dataloader, loss_fn, optimizer, scheduler
):

    model.to(device)
    model.train()

    epoch_loss = 0.0
    print_freq = 0

    for step, batch in enumerate(train_dataloader):

        model.zero_grad()

        label = batch.pop(-1)

        print("label")
        print(label)

        concepts_batch, property_batch = train_dataset.add_context(batch)

        if print_freq < 1:
            log.info(f"concepts_batch : {concepts_batch}")
            log.info(f"property_batch : {property_batch}")
            print_freq += 1

        ids_dict = train_dataset.tokenize(concepts_batch, property_batch)

        (
            concept_inp_id,
            concept_attention_mask,
            concept_token_type_id,
            property_input_id,
            property_attention_mask,
            property_token_type_id,
        ) = [val.to(device) for _, val in ids_dict.items()]

        concept_embedding, property_embedding, logits = model(
            concept_input_id=concept_inp_id,
            concept_attention_mask=concept_attention_mask,
            concept_token_type_id=concept_token_type_id,
            property_input_id=property_input_id,
            property_attention_mask=property_attention_mask,
            property_token_type_id=property_token_type_id,
        )

        logits = (
            (concept_embedding * property_embedding)
            .sum(-1)
            .reshape(concept_embedding.shape[0], 1)
        )

        batch_loss = loss_fn(logits, label.reshape_as(logits).float().to(device))

        epoch_loss += batch_loss.item()

        batch_loss.backward()
        torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if step % 100 == 0 and not step == 0:

            batch_labels = label.reshape(-1, 1).detach().cpu().numpy()

            batch_logits = (
                torch.round(torch.sigmoid(logits)).reshape(-1, 1).detach().cpu().numpy()
            )

            batch_scores = compute_scores(batch_labels, batch_logits)

            log.info(
                f"Batch {step} of {len(train_dataloader)} ----> Batch Loss : {batch_loss}, Batch Binary F1 {batch_scores.get('binary_f1')}"
            )
            print(flush=True)

    avg_epoch_loss = epoch_loss / len(train_dataloader)

    return avg_epoch_loss


def evaluate(model, valid_dataset, valid_dataloader, loss_fn, device):

    model.eval()

    val_loss = 0.0
    val_logits, val_label = [], []
    print_freq = 0

    for step, batch in enumerate(valid_dataloader):

        label = batch.pop(-1)

        concepts_batch, property_batch = valid_dataset.add_context(batch)

        if print_freq < 1:
            log.info(f"concepts_batch : {concepts_batch}")
            log.info(f"property_batch : {property_batch}")
            print_freq += 1

        ids_dict = valid_dataset.tokenize(concepts_batch, property_batch)

        (
            concept_inp_id,
            concept_attention_mask,
            concept_token_type_id,
            property_input_id,
            property_attention_mask,
            property_token_type_id,
        ) = [val.to(device) for _, val in ids_dict.items()]

        with torch.no_grad():

            concept_embedding, property_embedding, logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                concept_token_type_id=concept_token_type_id,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
                property_token_type_id=property_token_type_id,
            )

        batch_loss = loss_fn(logits, label.reshape_as(logits).float().to(device))

        val_loss += batch_loss.item()
        torch.cuda.empty_cache()

        val_logits.extend(logits)
        val_label.extend(label)

    epoch_logits = (
        torch.round(torch.sigmoid(torch.vstack(val_logits)))
        .reshape(-1, 1)
        .detach()
        .cpu()
        .numpy()
    )
    epoch_labels = torch.vstack(val_label).reshape(-1, 1).detach().cpu().numpy()

    scores = compute_scores(epoch_labels, epoch_logits)

    avg_val_loss = val_loss / len(valid_dataloader)

    return avg_val_loss, scores


def train(model, config, train_df, valid_df=None):

    log.info("Initialising datasets...")
    # dataset_params, dataset_type, data_df=None

    train_dataset, train_dataloader = mcrae_dataset_and_dataloader(
        dataset_params=config.get("dataset_params"),
        dataset_type="train",
        data_df=train_df,
    )

    valid_dataset, valid_dataloader = mcrae_dataset_and_dataloader(
        dataset_params=config.get("dataset_params"),
        dataset_type="valid",
        data_df=valid_df,
    )

    # -------------------- Preparation for training  ------------------- #

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=config["training_params"].get("lr"))
    total_training_steps = len(train_dataloader) * config["training_params"].get(
        "max_epochs"
    )

    # warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    # log.info(f"Warmup-steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training_params"].get("num_warmup_steps"),
        num_training_steps=total_training_steps,
    )

    best_val_f1 = 0.0
    start_epoch = 1

    epoch_count = []
    train_losses = []
    valid_losses = []

    log.info(f"Training the concept property model on {device}")

    patience_counter = 0

    for epoch in range(start_epoch, config["training_params"].get("max_epochs") + 1):

        log.info(f"  Epoch {epoch} of {config['training_params'].get('max_epochs')}")
        print("\n", flush=True)

        train_loss = train_single_epoch(
            model=model,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        log.info(f"Train Epoch {epoch} finished !!")
        log.info(f"  Average Train Loss: {train_loss}")

        if valid_df is None:
            model_save_path = os.path.join(
                config["training_params"].get("export_path"),
                config["model_params"].get("model_name"),
            )

            log.info(f"patience_counter : {patience_counter}")
            log.info(f"best_model_path : {model_save_path}")

            torch.save(
                model.state_dict(), model_save_path,
            )

            log.info(f"The model is saved in : {model_save_path}")

        # ----------------------------------------------#
        # ----------------------------------------------#
        # ---------------Validation---------------------#
        # ----------------------------------------------#
        # ----------------------------------------------#

        if valid_df is not None:
            log.info(f"Running Validation ....")
            print(flush=True)

            valid_loss, valid_scores = evaluate(
                model=model,
                valid_dataset=valid_dataset,
                valid_dataloader=valid_dataloader,
                loss_fn=loss_fn,
                device=device,
            )

            epoch_count.append(epoch)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            log.info(f"  Average validation Loss: {valid_loss}")
            print(flush=True)

            val_binary_f1 = valid_scores.get("binary_f1")

            if val_binary_f1 < best_val_f1:
                patience_counter += 1
            else:
                patience_counter = 0
                best_val_f1 = val_binary_f1

                best_model_path = os.path.join(
                    config["training_params"].get("export_path"),
                    config["model_params"].get("model_name"),
                )

                log.info(f"patience_counter : {patience_counter}")
                log.info(f"best_model_path : {best_model_path}")

                torch.save(
                    model.state_dict(), best_model_path,
                )

                log.info(f"Best model at epoch: {epoch}, Binary F1: {val_binary_f1}")
                log.info(f"The model is saved in : {best_model_path}")

            log.info("Validation Scores")
            log.info(f" Best Validation F1 yet : {best_val_f1}")

            for key, value in valid_scores.items():
                log.info(f"{key} : {value}")

            print(flush=True)

            print("train_losses", flush=True)
            print(train_losses, flush=True)
            print("valid_losses", flush=True)
            print(valid_losses, flush=True)

            if patience_counter >= config["training_params"].get(
                "early_stopping_patience"
            ):
                log.info(
                    f"Early Stopping ---> Maximum Patience - {config['training_params'].get('early_stopping_patience')} Reached !!"
                )
                break

            print(flush=True)


def cross_validation(model, config, concept_property_df, label_df):

    skf = StratifiedKFold(n_splits=5)

    for fold_num, (train_index, test_index) in enumerate(
        skf.split(concept_property_df, label_df)
    ):

        (concept_property_train_fold, concept_property_valid_fold,) = (
            concept_property_df.iloc[train_index],
            concept_property_df.iloc[test_index],
        )

        label_train_fold, label_valid_fold = (
            label_df.iloc[train_index],
            label_df.iloc[test_index],
        )

        train_df = pd.concat((concept_property_train_fold, label_train_fold), axis=1)
        valid_df = pd.concat((concept_property_valid_fold, label_valid_fold), axis=1)

        log.info(f"Running fold  : {fold_num + 1} of {skf.n_splits}")

        log.info(f"Total concept_property_df shape : {concept_property_df.shape}")
        log.info(f"Total label_df shape : {label_df.shape}")

        log.info("After Validation Split")
        log.info(
            f"concept_property_train_fold.shape : {concept_property_train_fold.shape}"
        )
        log.info(f"label_train_fold.shape : {label_train_fold.shape}")

        log.info(
            f"concept_property_valid_fold.shape : {concept_property_valid_fold.shape}"
        )

        log.info(f"label_valid_fold.shape : {label_valid_fold.shape}")

        log.info(f"Initialising training with fold : {fold_num + 1}")

        train(model, config, train_df, valid_df)


def test_best_model(config):

    log.info(f"\n {'*' * 50}")
    log.info(f"Testing the fine tuned model")

    model = create_model(config.get("model_params"))

    best_model_path = os.path.join(
        config["training_params"].get("export_path"),
        config["model_params"].get("model_name"),
    )

    log.info(f"Testing the best model : {best_model_path}")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    test_dataset, test_dataloader = mcrae_dataset_and_dataloader(
        dataset_params=config.get("dataset_params"), dataset_type="test", data_df=None
    )

    label = test_dataset.label
    all_test_preds = []

    for step, batch in enumerate(test_dataloader):

        concepts_batch, property_batch = test_dataset.add_context(batch)

        ids_dict = test_dataset.tokenize(concepts_batch, property_batch)

        (
            concept_inp_id,
            concept_attention_mask,
            concept_token_type_id,
            property_input_id,
            property_attention_mask,
            property_token_type_id,
        ) = [val.to(device) for _, val in ids_dict.items()]

        with torch.no_grad():

            concept_embedding, property_embedding, logits = model(
                concept_input_id=concept_inp_id,
                concept_attention_mask=concept_attention_mask,
                concept_token_type_id=concept_token_type_id,
                property_input_id=property_input_id,
                property_attention_mask=property_attention_mask,
                property_token_type_id=property_token_type_id,
            )

        preds = torch.round(torch.sigmoid(logits))
        all_test_preds.extend(preds.detach().cpu().numpy().flatten())

    test_scores = compute_scores(label, np.asarray(all_test_preds))

    log.info(f"Test Metrices")
    log.info(f"Test labels shape: {label.shape}")
    log.info(f"Test Preds shape: {np.asarray(all_test_preds).shape}")

    for key, value in test_scores.items():
        log.info(f"{key} : {value}")
    print(flush=True)


if __name__ == "__main__":

    set_seed(12345)

    parser = ArgumentParser(description="Fine tune configuration")

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

    # model = load_pretrained_model(config)
    model = create_model(config["model_params"])
    log.info(f"The following model is trained")
    log.info(model)

    total_params, trainable_params = count_parameters(model)
    log.info(f"The total number of parameters in the model : {total_params}")
    log.info(f"Trainable parameters in the model : {trainable_params}")

    concept_property_df, label_df = read_data(config["dataset_params"])
    assert concept_property_df.shape[0] == label_df.shape[0]

    if config["training_params"].get("do_cv"):
        cross_validation(model, config, concept_property_df, label_df)
    else:
        train_df = pd.concat((concept_property_df, label_df), axis=1)
        train(model, config, train_df)

    test_best_model(config)

