import logging
import torch
import os
import torch.nn as nn

from argparse import ArgumentParser
from utils.functions import (
    set_seed,
    read_config,
    load_pretrained_model,
    compute_scores,
    mcrae_dataset_and_dataloader,
)
from transformers import AdamW, get_linear_schedule_with_warmup


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

        batch_loss = loss_fn(logits, label.reshape_as(logits).to(device))

        epoch_loss += batch_loss.item()

        batch_loss.backward()
        torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if step % 100 == 0 and not step == 0:

            batch_labels = batch_labels.reshape(-1, 1).detach().cpu().numpy()

            batch_logits = (
                torch.round(torch.sigmoid(batch_logits))
                .reshape(-1, 1)
                .detach()
                .cpu()
                .numpy()
            )

            batch_scores = compute_scores(batch_labels, batch_logits)

            log.info(
                f"Batch {step} of {len(train_dataloader)} ----> Batch Loss : {batch_loss}, Batch Binary F1 {batch_scores.get('binary_f1')}"
            )
            print(flush=True)

    avg_epoch_loss = epoch_loss / len(train_dataloader)

    return avg_epoch_loss


def train(model, config):

    log.info("Initialising datasets...")

    train_dataset, train_dataloader = mcrae_dataset_and_dataloader(
        config.get("dataset_params"), dataset_type="train"
    )

    # valid_dataset, valid_dataloader = create_dataset_and_dataloader(
    #     config.get("dataset_params"), dataset_type="valid"
    # )

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

        # log.info(f"Running Validation ....")
        # print(flush=True)

        # valid_loss, valid_scores = evaluate(
        #     model=model,
        #     valid_dataset=valid_dataset,
        #     valid_dataloader=valid_dataloader,
        #     loss_fn=loss_fn,
        #     device=device,
        # )

        # epoch_count.append(epoch)
        # train_losses.append(train_loss)
        # valid_losses.append(valid_loss)

        # log.info(f"  Average validation Loss: {valid_loss}")
        # print(flush=True)

        # val_binary_f1 = valid_scores.get("binary_f1")

        # if val_binary_f1 < best_val_f1:
        #     patience_counter += 1
        # else:
        #     patience_counter = 0
        #     best_val_f1 = val_binary_f1

        #     best_model_path = os.path.join(
        #         config["training_params"].get("export_path"),
        #         config["model_params"].get("model_name"),
        #     )

        #     log.info(f"patience_counter : {patience_counter}")
        #     log.info(f"best_model_path : {best_model_path}")

        #     torch.save(
        #         model.state_dict(), best_model_path,
        #     )

        #     log.info(f"Best model at epoch: {epoch}, Binary F1: {val_binary_f1}")
        #     log.info(f"The model is saved in : {best_model_path}")

        # log.info("Validation Scores")
        # log.info(f" Best Validation F1 yet : {best_val_f1}")

        # for key, value in valid_scores.items():
        #     log.info(f"{key} : {value}")

        # print(flush=True)

        # print("train_losses", flush=True)
        # print(train_losses, flush=True)
        # print("valid_losses", flush=True)
        # print(valid_losses, flush=True)

        # if patience_counter >= config["training_params"].get("early_stopping_patience"):
        #     log.info(
        #         f"Early Stopping ---> Maximum Patience - {config['training_params'].get('early_stopping_patience')} Reached !!"
        #     )
        #     break

        # print(flush=True)


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

    model = load_pretrained_model(config)

    train(model, config)
