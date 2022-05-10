#!/usr/bin/env python
# coding: utf-8

# from transformers import BertModel, BertTokenizer
# model = BertModel.from_pretrained("bert-large-uncased")
# model.save_pretrained("/home/amitgajbhiye/cardiff_work/100k_data_experiments/bert_large_uncased_pretrained/model/")
# tok = BertTokenizer.from_pretrained("bert-large-uncased")
# tok.save_pretrained("/home/amitgajbhiye/cardiff_work/100k_data_experiments/bert_large_uncased_pretrained/tokenizer/")

# In[1]:


import numpy as np
import pandas as pd
import os

import torch
import pickle
import nltk

from model.concept_property_model import ConceptPropertyModel
from utils.functions import create_model
from utils.functions import load_pretrained_model
from utils.functions import read_config
from utils.functions import mcrae_dataset_and_dataloader

from sklearn.neighbors import NearestNeighbors
from collections import Counter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# assert os.environ["CONDA_DEFAULT_ENV"] == "gvenv", "Activate 'gvenv' conda environment"

print(f"Device Name : {device}")
print(f"Conda Environment Name : {os.environ['CONDA_DEFAULT_ENV']}")


# In[2]:


def pos_tagger(x):

    tokens = nltk.word_tokenize(x)
    # print ("tokens :", tokens)
    # print ("pos tags :", nltk.pos_tag(tokens))
    return nltk.pos_tag(tokens)


# In[3]:


def tag_adj(pos_tag_list):

    tags = [word_tag[1] for word_tag in pos_tag_list]
    # print ("tags :", tags)
    # print ([tag == "JJ" for tag in tags])
    # print (all([tag == "JJ" for tag in tags]))

    if all([tag == "JJ" for tag in tags]):
        # print (f"Returning True : {tags}")
        return True
    else:
        # print (f"Returning False : {tags}")
        return False


# In[ ]:


# Function to count properties

local_files_list = [
    "data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv",
    "data/train_data/500k_MSCG/gkb_prop_500k_train.tsv",
]

prefix_adj_local_files_list = [
    "data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv"
]

hawk_files_list = [
    "/scratch/c.scmag3/biencoder_concept_property/data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv",
    "/scratch/c.scmag3/biencoder_concept_property/data/train_data/500k_MSCG/gkb_prop_500k_train.tsv",
]


def read_con_prop_data(files_list):

    df_list = []
    for i, file in enumerate(files_list):
        df_list.append(pd.read_csv(file, sep="\t", names=["concept", "property"]))

    print(f"Shapes of the DF read : {[df.shape for df in df_list]}")

    df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Columns in concatenated DF : {df.columns}")
    print(f"Concatenated DF shape : {df.shape}")

    df.dropna(axis=0, how="any", inplace=True)
    df.drop_duplicates(subset=["concept", "property"], keep="first", inplace=True)

    df["prop_count"] = -1

    unique_property = df["property"].unique()

    print("num_unique_property :", unique_property.shape, flush=True)
    print("unique_property :", unique_property, flush=True)

    df.set_index("property", inplace=True)

    for i, prop in enumerate(unique_property):
        df.loc[prop, "prop_count"] = df.loc[prop].shape[0]

    df.reset_index(inplace=True)

    df = df[["concept", "property", "prop_count"]]

    df.to_csv(
        "data/evaluation_data/nn_analysis/only_prefix_adj_with_prop_count.tsv",
        sep="\t",
        index=None,
        header=True,
    )


# read_con_prop_data(files_list=prefix_adj_local_files_list)


# In[ ]:


def get_top_k_properties(con_prop_file, pos_tag=False, cut_off=5):

    df = pd.read_csv(con_prop_file, sep="\t", header=0)

    # df.sort_values("prop_count", ascending=False, inplace=True)

    # print (f"DF sorted on prop count : {df}")

    df_prop_count_cut_off = df[df["prop_count"] >= cut_off]

    # df_prop_count_cut_off = df_prop_count_cut_off[0:2000]

    print(f"Dataframe with prop_count >= {cut_off} = {df_prop_count_cut_off.shape}")

    if pos_tag:
        df_prop_count_cut_off["pos_tag"] = df_prop_count_cut_off["property"].apply(
            pos_tagger
        )
        df_prop_count_cut_off["is_only_adj"] = df_prop_count_cut_off["pos_tag"].apply(
            tag_adj
        )
        df_prop_count_cut_off = df_prop_count_cut_off[
            df_prop_count_cut_off["is_only_adj"] == True
        ]

        adj_true_df = df_prop_count_cut_off[
            df_prop_count_cut_off["is_only_adj"] == True
        ]

        print(f"adj_true_df shape {adj_true_df.shape}")
        print(
            f"adj_true_df['property'].unique - len :",
            adj_true_df["property"].unique().shape,
        )

    df_prop_count_cut_off.to_csv(
        "data/evaluation_data/nn_analysis/df_with_tags.tsv", sep="\t", index=False
    )
    print(df_prop_count_cut_off)

    unique_properties = df_prop_count_cut_off["property"].unique()
    unique_properties = [
        x.strip().replace("(part)", "").replace(".", "") for x in unique_properties
    ]
    num_unique_properties = df_prop_count_cut_off["property"].unique().shape

    print(f"Number of unique properties in cut_off DF : {num_unique_properties}")
    # print (f"Unique Properties are : {unique_properties}")

    df_list = [("dummy", prop, 0) for prop in unique_properties]

    df_prop = pd.DataFrame.from_records(df_list)

    df_prop.to_csv(
        "data/evaluation_data/nn_analysis/adjs_prop_count_5_prefix_adj_plus_gkb_prop_with_prop_count.tsv",
        sep="\t",
        index=None,
        header=None,
    )


# get_top_k_properties("data/evaluation_data/nn_analysis/hd_data/prefix_adj_plus_gkb_prop_with_prop_count.tsv", pos_tag=True, cut_off=15)


# In[4]:


hd_vocab_file = "data/evaluation_data/nn_analysis/hd_data/1A.english.vocabulary.txt"
test_file = "data/evaluation_data/nn_analysis/hd_data/hd_concept_test.csv"


def preprocess_hd_data(vocab_file, test_concept_file):

    with open(vocab_file, "r") as f:
        lines = f.readlines()
        lines = [("dummy", prop.strip(), int(0)) for prop in lines]

    con_prop_vocab_df = pd.DataFrame.from_records(lines)

    con_prop_vocab_df.to_csv(
        "data/evaluation_data/nn_analysis/hd_data/properties_hd_vocab_con_prop.tsv",
        sep="\t",
        index=None,
    )

    test_concepts_df = pd.read_csv(test_concept_file, sep=",", header=0)
    print(f"Test Concepts DF shape : {test_concepts_df.shape}")

    test_cons_list = test_concepts_df["hypo"].unique()

    print(f"Num Test Concepts : {len(test_cons_list)}")

    test_con_prop_list = [(con.strip(), "dummy", int(0)) for con in test_cons_list]

    test_con_prop_df = pd.DataFrame.from_records(test_con_prop_list)

    test_con_prop_df.to_csv(
        "data/evaluation_data/nn_analysis/hd_data/concepts_hd_test_con_prop.tsv",
        sep="\t",
        index=None,
    )


# preprocess_hd_data (vocab_file = hd_vocab_file, test_concept_file= test_file)


# In[ ]:


# Loading the BERT Large Model for generating Property Embedding
# Here change the property test_file in config to the tsv file which contain the properties

local_prop_config_file_path = (
    "configs/nn_analysis/prop_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"
)
hawk_prop_config_file_path = "configs/nn_analysis/hawk_prop_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"

torch.cuda.empty_cache()

prop_config = read_config(hawk_prop_config_file_path)
prop_model = load_pretrained_model(prop_config)
prop_model.eval()
prop_model.to(device)
print("Property Model Loaded")


# In[ ]:


# Get the embeddings for property and concepts


def get_embedding(model, config):

    test_dataset, test_dataloader = mcrae_dataset_and_dataloader(
        dataset_params=config.get("dataset_params"), dataset_type="test", data_df=None,
    )

    con_list, con_emb, prop_list, prop_emb = [], [], [], []

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

            print()
            print(f"Concepts Data :", len(batch[0]))
            print(f"Concepts Data :", batch[0])
            print(f"concept_embedding.shape : {concept_embedding.shape}")

            print(f"Property Data :", len(batch[1]))
            print(f"Property Data :", batch[1])
            print(f"property_embedding.shape : {property_embedding.shape}")

            # con_vec = [(con, vec) for con, vec in zip (batch[0], concept_embedding)]
            # prop_vec = [(prop, vec) for prop, vec in zip(batch[1], property_embedding)]

            con_list.extend(batch[0])
            con_emb.extend(concept_embedding)

            prop_list.extend(batch[1])
            prop_emb.extend(property_embedding)

    con_emb = [x.cpu().numpy() for x in con_emb]
    prop_emb = [x.cpu().numpy() for x in prop_emb]

    return con_list, con_emb, prop_list, prop_emb


# In[ ]:


_, _, prop_list, prop_emb = get_embedding(prop_model, prop_config)


# In[ ]:


print(f"prop_list len - {len(prop_list)}, Property Emb Len - {len(prop_emb)}")


# In[ ]:


def transform(vecs):

    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []

    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm ** 2 - np.linalg.norm(v) ** 2)))

    return new_vecs


# In[ ]:


prop_trans = transform(prop_emb)
print(len(prop_trans))

# print (prop_list)


# In[ ]:


prop_name_emb_dict = {"prop_name_list": prop_list, "prop_transformed_emb": prop_trans}


# In[ ]:


print(f"prop_name_emb_dict : {prop_name_emb_dict}")


# In[ ]:


# Pickling the transformed property name list and their embeddings.
with open(
    "data/evaluation_data/nn_analysis/hd_data/hd_prop_name_emb.pickle", "wb"
) as f:
    pickle.dump(prop_name_emb_dict, f)


# In[ ]:


# In[ ]:


# Loading the model model to generate concept embeddings
# Here change the concept test file the file where the test (query) concepts are loaded

torch.cuda.empty_cache()

local_con_conf_file_path = (
    "configs/nn_analysis/con_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"
)
hawk_con_conf_file_path = "configs/nn_analysis/hawk_prop_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"

con_config = read_config(hawk_con_conf_file_path)
con_model = load_pretrained_model(con_config)
con_model.eval()
con_model.to(device)
print("Concept Model Loaded")


# In[ ]:


con_list, con_emb, _, _ = get_embedding(con_model, con_config)


# In[ ]:


print(f"con_list len - {len(con_list)}, con_emb Len - {len(con_emb)}")


# In[ ]:


con_trans = transform(con_emb)
assert len(con_list) == len(con_trans)
print(len(con_trans))


# In[ ]:


con_name_emb_dict = {"con_name_list": con_list, "con_transformed_emb": con_trans}


# In[ ]:


con_name_emb_dict


# In[ ]:


with open("data/evaluation_data/nn_analysis/hd_data/hd_con_name_emb.pickle", "wb") as f:
    pickle.dump(con_name_emb_dict, f)


# In[ ]:


with open(
    "data/evaluation_data/nn_analysis/hd_data/hd_con_name_emb.pickle", "rb"
) as con_emb, open(
    "data/evaluation_data/nn_analysis/hd_data/hd_prop_name_emb.pickle", "rb"
) as prop_emb:

    con_name_emb = pickle.load(con_emb)
    prop_name_emb = pickle.load(prop_emb)

print(con_name_emb.keys())
print(prop_name_emb.keys())


# In[ ]:


print(
    f'Number of Properties in the loaded prop pickel : {len(prop_name_emb.get("prop_name_list"))}'
)
print(
    f'Number of Properties Embedding in the loaded prop pickel : {len(prop_name_emb.get("prop_transformed_emb"))}'
)

print(
    f'Number of Concepts in the loaded con pickel : {len(con_name_emb.get("con_name_list"))}'
)
print(
    f'Number of Concepts Embedding in the loaded prop pickel : {len(con_name_emb.get("con_transformed_emb"))}'
)


# In[ ]:


num_nearest_neighbours = 15


# In[ ]:


# Learning Nearest Neighbours
nbrs = NearestNeighbors(n_neighbors=num_nearest_neighbours, algorithm="brute").fit(
    np.array(prop_name_emb.get("prop_transformed_emb"))
)


# In[ ]:


distances, indices = nbrs.kneighbors(np.array(con_name_emb.get("con_transformed_emb")))


# In[ ]:


# print (indices)


# In[ ]:


for idx, con in zip(indices, con_name_emb.get("con_name_list")):
    print(
        f"{con} : {[prop_name_emb.get('prop_name_list') [prop_id] for prop_id in idx]}\n"
    )


generated_hypernyms_file = "data/evaluation_data/nn_analysis/hd_data/hd_test_concepts_generated_hypernyms_file.txt"

with open(generated_hypernyms_file, "r") as file:
    for idx, con in zip(indices, con_name_emb.get("con_name_list")):
        file.write(
            f"{con} : {[prop_name_emb.get('prop_name_list') [prop_id] for prop_id in idx]}\n"
        )


# In[ ]:


# In[ ]:


# In[ ]:


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# In[ ]:


carrots = [
    "come in many varieties",
    "have similar taste",
    "cruciferous vegetables",
    "have strong smell",
    "contain essential nutrients",
    "green",
    "contain fatty acid",
    "high in fiber",
    "annual plants",
    "provide necessary nutrients",
]

carrot = [
    "fresh",
    "non-starchy",
    "edible",
    "cruciferous vegetable",
    "solid food",
    "delicious",
    "cruciferous",
    "fiber-rich",
    "uncooked",
    "naturally gluten free",
]


# In[ ]:


print(intersection(carrots, carrot))


# In[ ]:


scooters = [
    "have second gear",
    "capable of crashs",
    "mechanical devices",
    "located in new jerseys",
    "capable of jumps",
    "automotive products",
    "capable of slow traffic",
    "electrical devices",
    "have enough power",
    "electronic devices",
]

scooter = [
    "durable",
    "wearable and",
    "moveable",
    "stationary",
    "wearable",
    "easily repairable",
    "expendable",
    "inflatable",
    "assistive",
    "small",
]


# In[ ]:


print(intersection(scooters, scooter))


# In[ ]:


bananas = [
    "edible fruit",
    "have similar taste",
    "come in many varieties",
    "high in fiber",
    "have strong smell",
    "produce small fruit",
    "contain fatty acid",
    "come in many colors",
    "have green color",
    "contain essential nutrients",
]

banana = [
    "solanaceous",
    "fresh",
    "edible",
    "tropical",
    "fiber-rich",
    "commercially important",
    "seasonal fresh",
    "solid food",
    "seasonal",
    "edible fruit",
]


# In[ ]:


print(intersection(bananas, banana))


# In[ ]:

