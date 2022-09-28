import argparse
import logging
import os
import pickle
import pandas as pd

import torch

from utils.functions import (
    create_model,
    read_config,
    to_cpu,
    mcrae_dataset_and_dataloader,
    load_pretrained_model,
)


log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()

device = torch.device("cuda") if cuda_available else torch.device("cpu")


def preprocess_get_embedding_data(config):

    inference_params = config.get("inference_params")

    data_df = pd.read_csv(
        inference_params.get("con_property_file"), sep="\t", header=None,
    )

    num_columns = len(data_df.columns)
    log.info(f"Number of columns in input file : {num_columns}")

    input_data_type = inference_params["input_data_type"]

    if input_data_type == "concept" and num_columns == 1:

        log.info(f"Generating Embeddings for Concepts")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "concept"}, inplace=True)

        unique_concepts = data_df["concept"].unique()
        data_df = pd.DataFrame(unique_concepts, columns=["concept"])

        data_df["property"] = "dummy_property"

    elif input_data_type == "property" and num_columns == 1:

        log.info("Generating Embeddings for Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "property"}, inplace=True)

        unique_properties = data_df["property"].unique()
        data_df = pd.DataFrame(unique_properties, columns=["property"])

        data_df["concept"] = "dummy_concept"

    elif input_data_type == "concept_and_property" and num_columns == 2:

        log.info("Generating Embeddings for Concepts and Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "concept", 1: "property"}, inplace=True)
        data_df.drop_duplicates(inplace=True)
        data_df.dropna(inplace=True)

    else:
        raise Exception(
            f"Please Enter a Valid Input data type from: 'concept', 'property' or conncept and property. \
            Current 'input_data_type' is: {input_data_type}"
        )

    data_df["label"] = int(0)
    data_df = data_df[["concept", "property", "label"]]

    return data_df


def generate_embedings(config):

    inference_params = config.get("inference_params")
    input_data_type = inference_params["inference_params"]
    model_params = config.get("model_params")
    dataset_params = config.get("dataset_params")

    model = create_model(model_params)

    best_model_path = inference_params["pretrained_model_path"]

    if cuda_available:
        model.load_state_dict(torch.load(best_model_path))
    else:
        model.load_state_dict(
            torch.load(best_model_path, map_location=torch.device("cpu"))
        )

    model.eval()
    model.to(device)

    log.info(f"The model is loaded from :{best_model_path}")
    log.info(f"The model is loaded on : {device}")

    data_df = preprocess_get_embedding_data(config=config)

    dataset, dataloader = mcrae_dataset_and_dataloader(
        dataset_params, dataset_type="test", data_df=data_df
    )

    embeddings = []

    for step, batch in enumerate(dataloader):

        concepts_batch, property_batch = dataset.add_context(batch)

        ids_dict = dataset.tokenize(concepts_batch, property_batch)

        if dataset.hf_tokenizer_name in ("roberta-base", "roberta-large"):

            (
                concept_inp_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
            ) = [val.to(device) for _, val in ids_dict.items()]

            concept_token_type_id = None
            property_token_type_id = None

        else:
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

            # print("shape concept_pair_embedding: ", concept_pair_embedding.shape)
            # print("shape relation_embedding: ", relation_embedding.shape)

        if input_data_type == "concept":

            for con, con_embed in zip(batch[0], concept_embedding):
                embeddings.append((con, to_cpu(con_embed)))

        elif input_data_type == "property":

            for prop, prop_embed in zip(batch[1], property_embedding):
                embeddings.append((prop, to_cpu(prop_embed)))

        elif input_data_type == "concept_and_property":

            for con, prop, con_embed, prop_embed in zip(
                batch[0], batch[1], concept_embedding, property_embedding
            ):
                embeddings.append((con, prop, to_cpu(con_embed), to_cpu(prop_embed)))

    save_dir = inference_params["save_dir"]
    file_name = os.path.join(save_dir, f"{input_data_type}_embedding.pkl")

    with open(file_name, "wb") as pkl_file:
        pickle.dump(embeddings, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

    log.info("Finished Generating the Embeddings")
    log.info(f"Embeddings are saved in : {file_name}")
    log.info(f"{'*' * 20} Finished {'*' * 20}")

    # with open(file_name, "rb") as pkl_file:
    #     emb_from_pkl = pickle.load(pkl_file)


if __name__ == "__main__":

    log.info(f"\n {'*' * 50}")
    log.info(f"Generating the Concept Property Embeddings")

    parser = argparse.ArgumentParser(description="Concept Property Biencoder Model")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )

    args = parser.parse_args()

    log.info(f"Reading Configuration File: {args.config_file}")
    config = read_config(args.config_file)

    log.info("The embeddings are generated with the following configuration")

    log.info(f"\n {config} \n")

    generate_embedings(config)
