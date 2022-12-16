import argparse
import random
import logging
import os
import pickle
import pandas as pd
import numpy as np

import torch

from nltk.stem import WordNetLemmatizer
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


def match_multi_words(word1, word2):

    lemmatizer = WordNetLemmatizer()

    word1 = " ".join([lemmatizer.lemmatize(word) for word in word1.split()])
    word2 = " ".join([lemmatizer.lemmatize(word) for word in word2.split()])

    return word1 == word2


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
        + f"_concept_similar_{num_nearest_neighbours}_vocab_properties"
        + ".tsv"
    )

    with open(file_name, "w") as file:

        for con_idx, prop_idx in enumerate(con_indices):

            concept = concepts[con_idx]
            similar_properties = [properties[idx] for idx in prop_idx]

            similar_properties = [
                prop
                for prop in similar_properties
                if not match_multi_words(concept, prop)
            ]

            con_similar_prop_dict[concept] = similar_properties

            print(f"{concept} \t {similar_properties}\n")

            for prop in similar_properties:
                line = concept + "\t" + prop + "\n"
                file.write(line)

    log.info(f"Finished getting similar properties")


def get_predict_prop_similar_properties(
    input_file,
    con_similar_prop,
    prop_vocab_embed_pkl,
    predict_prop_embed_pkl,
    save_file,
    num_prop_conjuct=5,
):

    # je_filtered_con_prop_file = "siamese_concept_property/data/train_data/joint_encoder_property_conjuction_data/je_filtered_con_similar_vocab_properties.txt"
    # je_filtered_prop_embed_pkl = "/home/amitgajbhiye/cardiff_work/concept_property_embeddings/prop_vocab_500k_mscg_embeds_property_embeddings.pkl"
    # predict_prop_embed_pkl = "/home/amitgajbhiye/cardiff_work/concept_property_embeddings/predict_property_embeds_cnet_premium_property_embeddings.pkl"

    if os.path.isfile(input_file):
        with open(input_file, "rb") as pkl_file:
            input_df = pickle.load(pkl_file)

    elif isinstance(input_file):
        input_df = input_file.rename(
            columns={0: "concept", 1: "predict_property", 2: "label"}
        )

    print(input_df.head(n=20))
    log.info(input_df.head(n=20))

    input_concepts = input_df["concept"].unique()
    input_predict_props = input_df["predict_property"].unique()

    num_input_concepts = len(input_concepts)
    num_input_predict_props = len(input_predict_props)

    je_filtered_con_prop_df = pd.read_csv(
        con_similar_prop, sep="\t", names=["concept", "similar_property"]
    )

    with open(prop_vocab_embed_pkl, "rb") as prop_vocab:
        prop_vocab_embeds_dict = pickle.load(prop_vocab)

    with open(predict_prop_embed_pkl, "rb") as predict_prop:
        predict_prop_embeds_dict = pickle.load(predict_prop)

    print(
        f"JE Filtered Concept Similar Properties DF Shape: {je_filtered_con_prop_df.shape}",
        flush=True,
    )
    print(
        f"Unique Properties in JE Filtered Con Prop Df : {len(je_filtered_con_prop_df['similar_property'].unique())}",
        flush=True,
    )

    print(
        f"Whole Property Vocab Embeddings : {len(prop_vocab_embeds_dict.keys())}",
        flush=True,
    )

    print()
    print(f"Input DF Shape : {input_df.shape}", flush=True)
    print(f"#Unique input concepts : {num_input_concepts}", flush=True)
    print(f"#Unique input predict properties : {num_input_predict_props}", flush=True)

    all_data = []
    for idx, (concept, predict_property, label) in enumerate(
        zip(input_df["concept"], input_df["predict_property"], input_df["label"])
    ):

        print(f"Processing Concept : {concept}, {idx+1} / {num_input_concepts}")
        print(
            f"Concept, Predict Property, Label : {(concept, predict_property, label)}"
        )

        similar_props = (
            je_filtered_con_prop_df[je_filtered_con_prop_df["concept"] == concept][
                "similar_property"
            ]
            .unique()
            .tolist()
        )
        similar_props = [
            prop for prop in similar_props if not match_multi_words(predict_prop, prop)
        ]

        embed_predict_prop = predict_prop_embeds_dict[predict_prop]
        embed_similar_prop = [prop_vocab_embeds_dict[prop] for prop in similar_props]

        zero_embed_predict_prop = np.array(
            np.insert(embed_predict_prop, 0, float(0))
        ).reshape(1, -1)
        transformed_embed_similar_prop = np.array(transform(embed_similar_prop))

        if len(similar_props) >= num_prop_conjuct:
            num_nearest_neighbours = num_prop_conjuct
        else:
            num_nearest_neighbours = len(similar_props)

        predict_prop_similar_props = NearestNeighbors(
            n_neighbors=num_nearest_neighbours, algorithm="brute"
        ).fit(transformed_embed_similar_prop)

        (
            similar_prop_distances,
            similar_prop_indices,
        ) = predict_prop_similar_props.kneighbors(zero_embed_predict_prop)

        similar_prop_indices = np.squeeze(similar_prop_indices)
        similar_properties = [similar_props[idx] for idx in similar_prop_indices]

        conjuct_similar_props = ", ".join(similar_properties)

        print(f"Concept : {concept}", flush=True)
        print(f"Predict Property : {predict_prop}", flush=True)
        print(f"Predict Property Similar Properties", flush=True)
        print(similar_properties, flush=True)
        print(f"Conjuct Similar Props", flush=True)
        print(conjuct_similar_props, flush=True)
        print("*" * 30, flush=True)
        print(flush=True)

        all_data.append([concept, conjuct_similar_props, predict_prop, label])

    df_all_data = pd.DataFrame.from_records(all_data)
    df_all_data.to_csv(save_file, sep="\t", header=None, index=None)


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

    log.info("The program is run with following configuration")
    log.info(f"{config} \n")

    inference_params = config.get("inference_params")

    get_con_prop_embeds = inference_params["get_con_prop_embeds"]
    get_con_similar_properties = inference_params["get_concept_similar_properties"]
    get_predict_prop_similar_props = inference_params["get_predict_prop_similar_props"]

    log.info(
        f"Get Concept, Property or Concept and Property Embedings : {get_con_prop_embeds}"
    )
    log.info(f"Get Concept Similar Vocab Properties  : {get_con_similar_properties} ")
    log.info(
        f"Get Predict Similar JE Filtered Properties  : {get_predict_prop_similar_props} "
    )

    if get_con_prop_embeds:

        input_data_type = inference_params["input_data_type"]

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

    if get_predict_prop_similar_props:

        pretrain_data = inference_params["pretrain_data"]
        finetune_data = inference_params["finetune_data"]
        num_prop_conjuct = inference_params["num_prop_conjuct"]

        predict_property_embed_pkl = inference_params.get("predict_property_embed_pkl")
        vocab_property_embed_pkl = inference_params.get("vocab_property_embed_pkl")
        concept_similar_prop_file = inference_params.get("concept_similar_prop_file")
        save_dir = inference_params["save_dir"]

        print()
        print(f"pretrain_data : {pretrain_data}")
        print(f"finetune_data : {finetune_data}")
        print(f"num_prop_conjuct : {num_prop_conjuct}")
        print(f"predict_property_embed_pkl : {predict_property_embed_pkl}")
        print(f"vocab_property_embed_pkl : {vocab_property_embed_pkl}")
        print(f"concept_similar_prop_file : {concept_similar_prop_file}")
        print(f"save_dir : {save_dir}")
        print()

        if pretrain_data:

            input_file_base_path = inference_params["pretrain_data"]

            train_file = os.path.join(input_file_base_path, "train_file_name---------")
            save_train_file = os.path.join(
                input_file_base_path, "save_train_file_name-------"
            )

            get_predict_prop_similar_properties(
                input_file=train_file,
                con_similar_prop=concept_similar_prop_file,
                prop_vocab_embed_pkl=vocab_property_embed_pkl,
                predict_prop_embed_pkl=predict_property_embed_pkl,
                save_file=save_train_file,
                num_prop_conjuct=num_prop_conjuct,
            )

            valid_file = os.path.join(
                input_file_base_path, "valid_file_name----------------"
            )
            save_valid_file = os.path.join(
                input_file_base_path, "valid_save_valid_file_name---------"
            )

            get_predict_prop_similar_properties(
                input_file=valid_file,
                con_similar_prop=concept_similar_prop_file,
                prop_vocab_embed_pkl=vocab_property_embed_pkl,
                predict_prop_embed_pkl=predict_property_embed_pkl,
                save_file=save_valid_file,
                num_prop_conjuct=num_prop_conjuct,
            )

        elif finetune_data:

            split_type = inference_params["split_type"]

            if split_type == "property_split":
                num_folds = 5

                train_file_suffix = "train_prop_split_con_prop.pkl"
                test_file_suffix = "test_prop_split_con_prop.pkl"

                input_file_base_path = inference_params["input_file_base_path"]

            elif split_type == "concept_property_split":
                num_folds = 9

                train_file_suffix = "------------"
                test_file_suffix = "------------"
                save_file_suffix = "------------"

            for fold_num in range(num_folds):

                train_file = os.path.join(
                    input_file_base_path, f"{fold_num}_{train_file_suffix}"
                )
                test_file = os.path.join(
                    input_file_base_path, f"{fold_num}_{test_file_suffix}"
                )

                with open(train_file, "rb") as train_pkl, open(
                    test_file, "rb"
                ) as test_pkl:
                    train_df = pickle.load(train_pkl)
                    test_df = pickle.load(test_pkl)

                train_save_file_name = os.path.join(
                    save_dir, f"{fold_num}_train_prop_conj_prop_split.tsv"
                )
                test_save_file_name = os.path.join(
                    save_dir, f"{fold_num}_test_prop_conj_prop_split.tsv"
                )

                log.info(f"Fold Number : {fold_num}")
                log.info(f"Train File : {train_file}")
                log.info(f"Test FIle : {test_file}")

                print(f"Fold Number : {fold_num}")
                print(f"Train File : {train_file}")
                print(f"Test FIle : {test_file}")

                get_predict_prop_similar_properties(
                    input_file=train_file,
                    con_similar_prop=concept_similar_prop_file,
                    prop_vocab_embed_pkl=vocab_property_embed_pkl,
                    predict_prop_embed_pkl=predict_property_embed_pkl,
                    save_file=train_save_file_name,
                    num_prop_conjuct=num_prop_conjuct,
                )

                get_predict_prop_similar_properties(
                    input_file=test_file,
                    con_similar_prop=concept_similar_prop_file,
                    prop_vocab_embed_pkl=vocab_property_embed_pkl,
                    predict_prop_embed_pkl=predict_property_embed_pkl,
                    save_file=test_save_file_name,
                    num_prop_conjuct=num_prop_conjuct,
                )