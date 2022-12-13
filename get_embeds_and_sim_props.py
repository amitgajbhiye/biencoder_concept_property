import argparse
import random
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

    log.info(f"Input Data Type : {input_data_type}")

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
        data_df["label"] = int(0)

    elif input_data_type == "property" and num_columns == 1:

        log.info("Generating Embeddings for Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "property"}, inplace=True)

        unique_properties = data_df["property"].unique()
        data_df = pd.DataFrame(unique_properties, columns=["property"])

        data_df["concept"] = "dummy_concept"
        data_df["label"] = int(0)

    elif input_data_type == "concept_and_property" and num_columns == 3:

        log.info("Generating Embeddings for Concepts and Properties")
        log.info(f"Number of records : {data_df.shape[0]}")

        data_df.rename(columns={0: "concept", 1: "property", 2: "label"}, inplace=True)

    else:
        raise Exception(
            f"Please Enter a Valid Input data type from: 'concept', 'property' or conncept_and_property. \
            Current 'input_data_type' is: {input_data_type}"
        )

    data_df = data_df[["concept", "property", "label"]]

    log.info(f"Final Data Df")
    log.info(data_df.head(n=20))

    return data_df


def generate_embeddings(config):

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
                # else:
                # log.info(f"Concept : {con} is already in dictionary !!")

            for prop, prop_embed in zip(batch[1], property_embedding):
                if prop not in prop_embedding:
                    prop_embedding[prop] = to_cpu(prop_embed)
                # else:
                # log.info(f"Property : {prop} is already in dictionary !!")

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

        con_file_name = dataset_params["dataset_name"] + "_concept_embeddings.pkl"
        prop_file_name = dataset_params["dataset_name"] + "_property_embeddings.pkl"

        con_embedding_save_file_name = os.path.join(save_dir, con_file_name)
        prop_embedding_save_file_name = os.path.join(save_dir, prop_file_name)

        with open(con_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(con_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        with open(prop_embedding_save_file_name, "wb") as pkl_file:
            pickle.dump(prop_embedding, pkl_file, protocol=pickle.DEFAULT_PROTOCOL)

        log.info(f"{'*' * 20} Finished {'*' * 20}")
        log.info("Finished Generating the Concept and Property Embeddings")
        log.info(f"Concept Embeddings are saved in : {con_embedding_save_file_name}")
        log.info(f"Property Embeddings are saved in : {prop_embedding_save_file_name}")
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

    log.info(f"Getting Concept Similar Properties ....")

    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    dataset_params = config.get("dataset_params")
    save_dir = inference_params["save_dir"]

    log.info(f"Input Data Type : {input_data_type}")

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
    log.info(f"Learning {num_nearest_neighbours} neighbours !!")

    con_similar_properties = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_prop_embeds))

    con_distances, con_indices = con_similar_properties.kneighbors(
        np.array(zero_con_embeds)
    )

    log.info(f"con_distances shape : {con_distances.shape}")
    log.info(f"con_indices shape : {con_indices.shape}")

    con_similar_prop_dict = {}
    file_name = (
        os.path.join(save_dir, dataset_params["dataset_name"])
        + f"concept_similar_{num_nearest_neighbours}_properties"
        + ".tsv"
    )

    with open(file_name, "w") as file:

        for con_idx, prop_idx in enumerate(con_indices):

            concept = concepts[con_idx]
            similar_properties = [properties[idx] for idx in prop_idx]

            con_similar_prop_dict[concept] = similar_properties

            print(f"{concept} \t {similar_properties}\n")

            for prop in similar_properties:
                line = concept + "\t" + prop + "\n"
                file.write(line)

    log.info(f"Finished getting similar properties")


def get_predict_prop_similar_vocab_properties(
    config, predict_property_embed_pkl_file, vocab_property_embed_pkl_file
):

    log.info(f"Getting Predict Properti Similar Vocab Properties")

    inference_params = config.get("inference_params")
    input_data_type = inference_params["input_data_type"]

    dataset_params = config.get("dataset_params")
    save_dir = inference_params["save_dir"]

    log.info(f"Input Data Type : {input_data_type}")

    with open(predict_property_embed_pkl_file, "rb") as predict_prop_pkl_file, open(
        vocab_property_embed_pkl_file, "rb"
    ) as vocab_prop_pkl_file:

        predict_prop_dict = pickle.load(predict_prop_pkl_file)
        vocab_prop_dict = pickle.load(vocab_prop_pkl_file)

    predict_props = list(predict_prop_dict.keys())
    predict_props_embeds = list(predict_prop_dict.values())

    zero_predict_props_embeds = np.array(
        [np.insert(l, 0, float(0)) for l in predict_props_embeds]
    )
    transformed_con_embeds = np.array(transform(predict_props_embeds))

    log.info(f"In get_predict_prop_similar_vocab_properties function")
    log.info(f"Number of predict properties : {len(predict_props)}")
    log.info(f"Length of predict prop Embeddings : {len(predict_props_embeds)}")
    log.info(f"Shape of zero_predict_props_embeds: {zero_predict_props_embeds.shape}")
    log.info(f"Shape of transformed_con_embeds : {transformed_con_embeds.shape}")

    vocab_props = list(vocab_prop_dict.keys())
    vocab_props_embeds = list(vocab_prop_dict.values())
    zero_vocab_prop_embeds = np.array([np.insert(l, 0, 0) for l in vocab_props_embeds])
    transformed_vocab_prop_embeds = np.array(transform(vocab_props_embeds))

    log.info(f"Number of Vocab Properties : {len(vocab_props)}")
    log.info(f"Length of Vocab Properties Embeddings : {len(vocab_props_embeds)}")
    log.info(f"Shape of zero_vocab_prop_embeds: {zero_vocab_prop_embeds.shape}")
    log.info(
        f"Shape of transformed_vocab_prop_embeds : {transformed_vocab_prop_embeds.shape}"
    )

    # Learning Nearest Neighbours
    num_nearest_neighbours = 15
    log.info(f"Learning {num_nearest_neighbours} neighbours !!")

    predict_prop_similar_voab_props = NearestNeighbors(
        n_neighbors=num_nearest_neighbours, algorithm="brute"
    ).fit(np.array(transformed_vocab_prop_embeds))

    (
        predict_prop_distances,
        predict_prop_indices,
    ) = predict_prop_similar_voab_props.kneighbors(np.array(zero_predict_props_embeds))

    log.info(f"predict_prop_distances shape : {predict_prop_distances.shape}")
    log.info(f"predict_prop_indices shape : {predict_prop_indices.shape}")

    # file_name = (
    #     os.path.join(save_dir, dataset_params["dataset_name"])
    #     + f"mcrae_predict_prop_similar_{num_nearest_neighbours}_vocab_properties"
    #     + ".tsv"
    # )

    predict_prop_similar_vocab_props = {}
    for predict_prop_idx, vocab_prop_idx in enumerate(predict_prop_indices):

        predict_prop = predict_props[predict_prop_idx]
        vocab_similar_properties = [vocab_props[idx] for idx in vocab_prop_idx]

        predict_prop_similar_vocab_props[predict_prop] = vocab_similar_properties

        print(f"{predict_prop} \t {vocab_similar_properties}\n")

    return predict_prop_similar_vocab_props


def create_property_conjuction_data_for_fine_tuning(
    predict_prop_similar_vocab_props, concept_similar_prop_file, data_df, save_path
):

    all_data = []

    concept_similar_prop_file = pd.read_csv(
        concept_similar_prop_file, sep="\t", names=["concept", "similar_prop"]
    )

    log.info(f"Input Data DF")
    log.info(f"{data_df.head(n=20)}")

    for concept, predict_prop, label in data_df.values:

        num_prop_conjuct = random.randint(3, 10)

        # Check logic here you are not mixing the concept similar properties and predict similar properties..

        if label == 1:

            conjuct_properties = predict_prop_similar_vocab_props[predict_prop]
            conjuct_properties = [
                prop
                for prop in conjuct_properties
                if prop.lower().strip() != predict_prop.lower().strip()
            ]
            conjuct_properties = conjuct_properties[0:num_prop_conjuct]

            conjuct_properties = ", ".join(conjuct_properties)

        elif label == 0:

            concept_data = concept_similar_prop_file[
                concept_similar_prop_file["concept"] == concept
            ]

            concept_data_props = concept_data["similar_prop"].unique().tolist()
            concept_data_props = [
                prop
                for prop in concept_data_props
                if prop.lower().strip() != predict_prop.lower().strip()
            ]

            conjuct_properties = random.sample(concept_data_props, num_prop_conjuct)

            conjuct_properties = ", ".join(conjuct_properties)

        all_data.append([concept, conjuct_properties, predict_prop, label])

    all_data_df = pd.DataFrame.from_records(
        all_data, columns=["concept", "conjuct_properties", "predict_prop", "label"]
    )

    all_data_df.to_csv(save_path, sep="\t", index=None, header=None)


if __name__ == "__main__":

    log.info(f"\n {'*' * 50}")
    log.info(f"Generating the Concept Property Embeddings")

    parser = argparse.ArgumentParser(
        description="Pretrained Concept Property Biencoder Model"
    )

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

    get_con_prop_embeds = inference_params["get_con_prop_embeds"]
    get_con_similar_properties = inference_params["get_concept_similar_properties"]
    get_predict_prop_sim_vocab_props = inference_params[
        "get_predict_prop_similar_vocab_properties"
    ]

    log.info(
        f"Get Concept, Property and Concept and Property Embedings : {get_con_prop_embeds}"
    )
    log.info(
        f"Do I need concept similar top properties aslso, Step 1 of Model : {get_con_similar_properties} "
    )

    if get_con_prop_embeds:

        assert input_data_type in (
            "concept",
            "property",
            "concept_and_property",
        ), "Please specify 'input_data_type' \
            from ('concept', 'property', 'concept_and_property')"

        if input_data_type == "concept":
            concept_pkl_file = generate_embeddings(config=config)

        elif input_data_type == "property":
            property_pkl_file = generate_embeddings(config=config)

        elif input_data_type == "concept_and_property":
            concept_pkl_file, property_pkl_file = generate_embeddings(config=config)

    if get_con_similar_properties:

        concept_embed_pkl = inference_params["concept_embed_pkl"]
        property_embed_pkl = inference_params["property_embed_pkl"]

        get_concept_similar_properties(
            config,
            concept_embedding_pkl_file=concept_embed_pkl,
            property_embedding_pkl_file=property_embed_pkl,
        )

    if get_predict_prop_sim_vocab_props:

        predict_property_embed_pkl_file = inference_params.get(
            "predict_property_embed_pkl_file"
        )
        vocab_property_embed_pkl_file = inference_params.get(
            "vocab_property_embed_pkl_file"
        )

        concept_similar_prop_file = inference_params.get("concept_similar_prop_file")

        predict_prop_similar_vocab_props = get_predict_prop_similar_vocab_properties(
            config=config,
            predict_property_embed_pkl_file=predict_property_embed_pkl_file,
            vocab_property_embed_pkl_file=vocab_property_embed_pkl_file,
        )

        save_path = inference_params.get("save_dir")

        num_folds = 5
        for fold_num in range(num_folds):

            base_prop_split_file_paths = (
                "data/evaluation_data/mcrae_prop_split_train_test_files"
            )
            train_file = os.path.join(
                base_prop_split_file_paths, f"{fold_num}_train_prop_split_con_prop.pkl"
            )
            test_file = os.path.join(
                base_prop_split_file_paths, f"{fold_num}_test_prop_split_con_prop.pkl"
            )

            with open(train_file, "rb") as train_pkl, open(test_file, "rb") as test_pkl:

                train_df = pickle.load(train_pkl)
                test_df = pickle.load(test_pkl)

            train_save_file_name = os.path.join(
                save_path, f"{fold_num}_prop_conj_train_prop_split_con_prop.pkl"
            )
            test_save_file_name = os.path.join(
                save_path, f"{fold_num}_prop_conj_test_prop_split_con_prop.pkl"
            )

            create_property_conjuction_data_for_fine_tuning(
                predict_prop_similar_vocab_props=predict_prop_similar_vocab_props,
                concept_similar_prop_file=concept_similar_prop_file,
                data_df=train_df,
                save_path=train_save_file_name,
            )

            create_property_conjuction_data_for_fine_tuning(
                predict_prop_similar_vocab_props=predict_prop_similar_vocab_props,
                concept_similar_prop_file=concept_similar_prop_file,
                data_df=test_df,
                save_path=test_save_file_name,
            )

