import argparse
from enum import unique
import logging
import os
import pickle
import pandas as pd
import numpy as np

import torch

from utils.functions import (
    create_model,
    read_config,
    to_cpu,
    mcrae_dataset_and_dataloader,
)
from sklearn.neighbors import NearestNeighbors


log = logging.getLogger(__name__)

cuda_available = torch.cuda.is_available()

device = torch.device("cuda") if cuda_available else torch.device("cpu")


def preprocess_get_embedding_data(config):

    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    if input_data_type == "concept":
        data_df = pd.read_csv(
            inference_params["concept_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
        )

    elif input_data_type == "property":
        data_df = pd.read_csv(
            inference_params["property_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
        )

    elif input_data_type == "concept_and_property":
        data_df = pd.read_csv(
            inference_params["concept_property_file"],
            sep="\t",
            header=None,
            keep_default_na=False,
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

    input_data_type = inference_params["input_data_type"]
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

    con_embedding, prop_embedding = {}, {}

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
                con_embedding[con] = to_cpu(con_embed)

        elif input_data_type == "property":

            for prop, prop_embed in zip(batch[1], property_embedding):
                prop_embedding[prop] = to_cpu(prop_embed)

        elif input_data_type == "concept_and_property":

            for con, con_embed in zip(batch[0], concept_embedding):
                if con not in con_embedding:
                    con_embedding[con] = to_cpu(con_embed)

            for prop, prop_embed in zip(batch[1], property_embedding):
                if prop not in prop_embedding:
                    prop_embedding[prop] = to_cpu(prop_embed)

    save_dir = inference_params["save_dir"]

    if input_data_type == "concept":
        file_name = dataset_params["dataset_name"] + "_concept_embeddings.pkl"
        embedding_save_file_name = os.path.join(save_dir, file_name)

        with open(embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept Embeddings")
        log.info(f"Concept Embeddings are saved in : {embedding_save_file_name}")
        log.info(f"{'*' * 40}")

        return embedding_save_file_name

    if input_data_type == "property":
        file_name = dataset_params["dataset_name"] + "_property_embeddings.pkl"
        embedding_save_file_name = os.path.join(save_dir, file_name)

        with open(embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Property Embeddings")
        log.info(f"Property Embeddings are saved in : {embedding_save_file_name}")
        log.info(f"{'*' * 40}")

        return embedding_save_file_name

    if input_data_type == "concept_and_property":

        con_file_name = dataset_params["dataset_name"] + "_con_embeddings.pkl"
        prop_file_name = dataset_params["dataset_name"] + "_prop_embeddings.pkl"

        con_embedding_save_file_name = os.path.join(save_dir, con_file_name)
        prop_embedding_save_file_name = os.path.join(save_dir, prop_file_name)

        with open(con_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        with open(prop_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept and Property Embeddings")
        log.info(
            f"Concept Property Embeddings are saved in : {con_embedding_save_file_name, prop_embedding_save_file_name}"
        )
        log.info(f"{'*' * 40}")

        return con_embedding_save_file_name, prop_embedding_save_file_name


######################################
def transform(vecs):

    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm ** 2 - np.linalg.norm(v) ** 2)))

    return new_vecs


def get_concept_similar_properties(
    config, concept_embedding_pkl_file, property_embedding_pkl_file
):

    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    dataset_params = config.get("dataset_params")
    save_dir = inference_params["save_dir"]

    if input_data_type == "concept_and_property":

        with open(concept_embedding_pkl_file, "rb") as con_pkl_file, open(
            property_embedding_pkl_file, "rb"
        ) as prop_pkl_file:

            con_dict = pickle.load(con_pkl_file)
            prop_dict = pickle.load(prop_pkl_file)

    concepts = list(con_dict.keys())
    con_embeds = list(con_dict.values())

    zero_con_embeds = np.array([np.insert(l, 0, float(0)) for l in con_embeds])
    transformed_con_embeds = np.array(transform(con_embeds))

    log.info(f"In get_concept_similar_properties function")
    log.info(f"Number of Concepts : {len(concepts)}")
    log.info(f"Length of Concepts Embeddings : {len(con_embeds)}")
    log.info(f"Shape of zero_con_embeds: {zero_con_embeds.shape}")
    log.info(f"Shape of transformed_con_embeds : {transformed_con_embeds.shape}")

    properties = list(prop_dict.keys())
    prop_embeds = list(prop_dict.values())
    zero_prop_embeds = np.array([np.insert(l, 0, 0) for l in prop_embeds])
    transformed_prop_embeds = np.array(transform(prop_embeds))

    log.info(f"Number of Properties : {len(properties)}")
    log.info(f"Length of Properties Embeddings : {len(prop_embeds)}")
    log.info(f"Shape of zero_prop_embeds: {zero_prop_embeds.shape}")
    log.info(f"Shape of transformed_prop_embeds : {transformed_prop_embeds.shape}")

    prop_dict_transform = {
        prop: trans for prop, trans in zip(properties, transformed_prop_embeds)
    }
    prop_dict_zero = {prop: trans for prop, trans in zip(properties, zero_prop_embeds)}

    # Learning Nearest Neighbours
    num_nearest_neighbours = 50

    con_similar_properties = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_prop_embeds))

    con_distances, con_indices = con_similar_properties.kneighbors(
        np.array(zero_con_embeds)
    )

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    con_similar_prop_dict = {}
    file_name = os.path.join(save_dir, dataset_params["dataset_name"])

    with open(file_name, "wb") as file:

        for con_idx, prop_idx in enumerate(con_indices):

            concept = concepts[con_idx]
            similar_properties = [properties[idx] for idx in prop_idx]

            con_similar_prop_dict[concept] = similar_properties

            print(f"{concept} - {similar_properties}")

            for prop in similar_properties:
                file.write(f"{concept}\t{prop}\n")

    log.info(f"Finished getting similar properties")

    # log.info(f"con_similar_prop_dict")
    # log.info(con_similar_prop_dict)

    # print(f"con_similar_prop_dict")
    # print(con_similar_prop_dict)

    # con_prop_file = inference_params["concept_property_file"]

    # con_prop_df = pd.read_csv(
    #     con_prop_file, sep="\t", names=["concept", "property"], header=None
    # )

    # unique_concepts = con_prop_df["concept"].unique()

    # all_data = []

    # for concept in unique_concepts:

    #     properties_of_concept = (
    #         con_prop_df[con_prop_df["concept"] == concept]["property"]
    #         .str.strip()
    #         .to_list()
    #     )

    #     top_k_con_similar_prop = con_similar_prop_dict[concept]

    #     top_k_con_similar_prop_transformed_embs = np.array(
    #         [prop_dict_transform[prop] for prop in top_k_con_similar_prop]
    #     )

    #     con_prop_embed = np.array(
    #         [prop_dict_zero[prop] for prop in properties_of_concept]
    #     )

    #     num_nearest_neighbours = 40

    #     prop_similar_properties = NearestNeighbors(
    #         n_neighbors=num_nearest_neighbours, algorithm="brute"
    #     ).fit(np.array(top_k_con_similar_prop_transformed_embs))

    #     prop_distances, prop_indices = prop_similar_properties.kneighbors(
    #         con_prop_embed
    #     )

    #     for idx, pro_idx in enumerate(prop_indices):

    #         prop_data = []

    #         predict_property = properties_of_concept[idx]

    #         predict_property_similar_properties = [
    #             top_k_con_similar_prop[idx] for idx in pro_idx
    #         ]

    #         print(f"{concept} - {predict_property}")

    #         for similar_prop in predict_property_similar_properties:
    #             # if predict_property != similar_prop:
    #             if similar_prop not in properties_of_concept:
    #                 prop_data.append(similar_prop)

    #             if len(prop_data) >= 10:
    #                 break

    #         # prop_data = ", ".join(prop_data)

    #         all_data.append((concept, prop_data, predict_property))

    #         print(
    #             f"Final Similar Properties for Concept Property Pair : {concept, predict_property}"
    #         )
    #         print(prop_data)

    #         print()

    #     print()

    # data_df = pd.DataFrame.from_records(
    #     all_data, columns=["concept", "similar_properties", "property"]
    # )

    # data_df.to_csv(
    #     "data/generate_embeddding_data/property_conjuction_data.tsv",
    #     sep="\t",
    #     header=None,
    #     index=None,
    # )

    # log.info(
    #     f"Extracted data saved in  : data/generate_embeddding_data/property_conjuction_data.tsv"
    # )


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

    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    assert input_data_type in (
        "concept",
        "property",
        "concept_and_property",
    ), "Please specify 'input_data_type' \
        from ('concept', 'property', 'concept_and_property')"

    concept_pkl_file = inference_params["concept_file"]
    property_pkl_file = inference_params["property_file"]

    get_concept_similar_properties(config, concept_pkl_file, property_pkl_file)

