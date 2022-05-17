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
import nltk
import pickle

from model.concept_property_model import ConceptPropertyModel
from utils.functions import create_model
from utils.functions import load_pretrained_model
from utils.functions import read_config
from utils.functions import mcrae_dataset_and_dataloader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# assert os.environ["CONDA_DEFAULT_ENV"] == "gvenv", "Activate 'gvenv' conda environment"

print (f"Device Name : {device}")
print (f"Conda Environment Name : {os.environ['CONDA_DEFAULT_ENV']}")


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
    
    if all ([tag == "JJ" for tag in tags]):
        # print (f"Returning True : {tags}")
        return True
    else:
        # print (f"Returning False : {tags}")
        return False


# In[4]:


def tag_noun(pos_tag_list):
    
    tags = [word_tag[1] for word_tag in pos_tag_list]
    
    if  tags[-1] in ("NN","NNS","NNPS"):
        # print (f"Returning True : {tags}")
        return True
    else:
        # print (f"Returning False : {tags}")
        return False


# In[ ]:


# Function to count properties

local_files_list = ["data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv",
             "data/train_data/500k_MSCG/gkb_prop_500k_train.tsv"]

prefix_adj_local_files_list = ["data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv"]

hawk_files_list = ["/scratch/c.scmag3/biencoder_concept_property/data/train_data/500k_MSCG/mscg_prefix_adj_41k_train.tsv",
             "/scratch/c.scmag3/biencoder_concept_property/data/train_data/500k_MSCG/gkb_prop_500k_train.tsv"]

def read_con_prop_data (files_list):
    
    df_list = []
    for i, file in enumerate(files_list):
        df_list.append(pd.read_csv(file, sep="\t", names=["concept", "property"]))
        
    print (f"Shapes of the DF read : {[df.shape for df in df_list]}")
    
    df = pd.concat(df_list, axis=0, ignore_index=True)
    print (f"Columns in concatenated DF : {df.columns}")
    print (f"Concatenated DF shape : {df.shape}")
    
    df.dropna(axis=0, how="any", inplace=True)
    df.drop_duplicates(subset=['concept', 'property'], keep="first", inplace=True)
    
    df["prop_count"] = -1
    
    unique_property = df["property"].unique()
    
    print ("num_unique_property :", unique_property.shape, flush=True)
    print ("unique_property :", unique_property, flush=True)
    
    df.set_index("property", inplace=True)
    
    for i, prop in enumerate(unique_property):
        df.loc[prop, "prop_count"] = df.loc[prop].shape[0]
    
    df.reset_index(inplace=True)
    
    df = df[["concept", "property", "prop_count"]]
    
    df.to_csv("data/evaluation_data/nn_analysis/only_prefix_adj_with_prop_count.tsv", sep='\t', index=None, header=True)

# read_con_prop_data(files_list=prefix_adj_local_files_list)


# In[ ]:


def get_top_k_properties(con_prop_file, pos_tag = False, cut_off = 5):
    
    df = pd.read_csv(con_prop_file, sep="\t", header=0)

    # df.sort_values("prop_count", ascending=False, inplace=True)

    # print (f"DF sorted on prop count : {df}")

    df_prop_count_cut_off = df[df["prop_count"] >= cut_off]
    
    # df_prop_count_cut_off = df_prop_count_cut_off[0:2000]

    print (f"Dataframe with prop_count >= {cut_off} = {df_prop_count_cut_off.shape}")
    
    if pos_tag:
        df_prop_count_cut_off["pos_tag"] = df_prop_count_cut_off["property"].apply(pos_tagger)
        
        df_prop_count_cut_off["is_only_adj"] = df_prop_count_cut_off["pos_tag"].apply(tag_adj)
        
        df_prop_count_cut_off["is_last_word_noun"] = df_prop_count_cut_off["pos_tag"].apply(tag_noun)
        
        
    df_prop_count_cut_off.to_csv("data/evaluation_data/nn_analysis/df_with_adj_and_noun_tags.tsv", sep="\t", index=False)
    print (df_prop_count_cut_off)
    
    
    df_is_only_adj = df_prop_count_cut_off[df_prop_count_cut_off["is_only_adj"] == True]
    
    adj_file_name = "data/evaluation_data/nn_analysis/prefix_plus_gkb_df_with_only_adj_properties.tsv"
    df_is_only_adj.to_csv(adj_file_name, sep="\t", index=None, header=None)
    
    unique_adj_prop = df_is_only_adj["property"].unique()
    unique_adj_prop = [x.strip().replace("(part)", "").replace(".", "") for x in unique_adj_prop]
    unique_adj_prop = [("dummy_con", prop, 0) for prop in unique_adj_prop]
    df_unique_adj_prop = pd.DataFrame.from_records(unique_adj_prop)
    
    df_adj_prop_file_name = "data/evaluation_data/nn_analysis/adj_properties.tsv"
    df_unique_adj_prop.to_csv(df_adj_prop_file_name, sep="\t", header=None, index=None)
    
    print (f"df_is_only_adj.shape : {df_is_only_adj.shape}")
    print ("Adjective Unique Property", len(unique_adj_prop))
    
    
    print ()
    df_last_word_noun = df_prop_count_cut_off[df_prop_count_cut_off["is_last_word_noun"] == True]
    print (f"df_last_word_noun.shape : {df_last_word_noun.shape}")
    
    noun_file_name = "data/evaluation_data/nn_analysis/prefix_plus_gkb_df_with_last_word_noun_properties.tsv"
    df_last_word_noun.to_csv(noun_file_name, sep="\t", index=None, header=None)
    
    unique_last_word_noun = df_last_word_noun["property"].unique()
    
    print ("unique_last_word_noun : ", len(unique_last_word_noun))
    
    unique_last_word_noun = [x.strip().replace("(part)", "").replace(".", "") for x in unique_last_word_noun]
    unique_last_word_noun = [("dummy_con", prop, 0) for prop in unique_last_word_noun]
    
    df_unique_last_word_noun = pd.DataFrame.from_records(unique_last_word_noun)
    df_unique_last_word_noun_file_name = "data/evaluation_data/nn_analysis/noun_properties.tsv"
    
    df_unique_last_word_noun.to_csv(df_unique_last_word_noun_file_name, sep="\t", header=None, index=None)

    print (f"df_unique_last_word_noun.shape : {df_unique_last_word_noun.shape}")
    print ("Noun Unique Property", len(unique_last_word_noun))
    
    
con_prop_file_with_counts = "data/evaluation_data/nn_analysis/prefix_adj_plus_gkb_prop_with_prop_count.tsv"
    
# get_top_k_properties(con_prop_file_with_counts, pos_tag=True, cut_off=10)


# In[ ]:





# In[6]:


# hd_vocab_file = "data/evaluation_data/nn_analysis/hd_data/1A.english.vocabulary.txt"
# test_file = "data/evaluation_data/nn_analysis/hd_data/hd_concept_test.csv"

music_hd_vocab = "data/evaluation_data/nn_analysis/music_hd/2B.music.vocabulary.txt"
music_hd_test = "data/evaluation_data/nn_analysis/music_hd/music__hd_concept_test.csv"

def preprocess_hd_data(vocab_file, test_concept_file):

    with open(vocab_file,  "r") as f:
        lines = f.readlines()
        lines = [("con_dummy", prop.strip(), int(0)) for prop in lines]
        
    con_prop_vocab_df = pd.DataFrame.from_records(lines)
    con_prop_vocab_df = pd.DataFrame.from_records(lines)[0:2500]
    
    con_prop_vocab_df.to_csv("data/evaluation_data/nn_analysis/music_hd/properties_music_hd_vocab_con_prop.tsv", sep="\t", index=None, header=None)
    
    
    test_concepts_df = pd.read_csv(test_concept_file, sep=",", header=0)
    print (f"Test Concepts DF shape : {test_concepts_df.shape}")
    print ("test_concepts_df")
    print (test_concepts_df)
    
    
    test_cons_list = test_concepts_df["hypo"].unique()
    # test_cons_list = test_concepts_df["hypo"].unique()[0:10]
    
    print (f"Num Test Concepts : {len(test_cons_list)}")
    
    test_con_prop_list = [(con.strip(), "prop_dummy", int(0)) for con in test_cons_list]
    
    test_con_prop_df  = pd.DataFrame.from_records(test_con_prop_list)
    
    test_con_prop_df.to_csv("data/evaluation_data/nn_analysis/music_hd/concepts_music_hd_test_con_prop.tsv", sep="\t", index=None, header=None)
    
    
# preprocess_hd_data (vocab_file = music_hd_vocab, test_concept_file = music_hd_test)


# con_prop_file = "data/evaluation_data/nn_analysis/prefix_adj_plus_gkb_prop_with_prop_count.tsv"
# 
# con_prop_df = pd.read_csv(con_prop_file, sep="\t", header=0)
# 
# print (con_prop_df)

# 
# file_cut_off_prop_count_10_unique = "data/evaluation_data/nn_analysis/cut_off_prop_count_10_unique.tsv"
# 
# cut_off_prop_count_10 = con_prop_df[con_prop_df["prop_count"] >= 10]
# 
# cut_off_prop_count_10.drop("prop_count", axis=1, inplace=True)
# 
# unique_prop_list = cut_off_prop_count_10["property"].unique()
# 
# print ("Unique Prop Count :", len(unique_prop_list))
# 
# unique_prop_list = [("dummy_con", prop.strip(), int(0)) for prop in unique_prop_list]
# 
# unique_prop_df = pd.DataFrame(unique_prop_list)
# 
# unique_prop_df.to_csv(file_cut_off_prop_count_10_unique, sep="\t", header=None, index=None)
# 
# 

# mc_train_file = "data/evaluation_data/extended_mcrae/train_mcrae.tsv"
# 
# train_df = pd.read_csv(mc_train_file, sep="\t", names=["concept", "property", "label"])
# 
# unique_train_con = train_df["concept"].unique()
# unique_train_con = [(con.strip(), "dummy_prop", int(0)) for con in unique_train_con]
# 
# unique_train_df = pd.DataFrame.from_records(unique_train_con)
# unique_train_df.to_csv("data/evaluation_data/nn_analysis/mcrae_unique_train_concepts.tsv", sep="\t", header=None, index=None)
# 
# 
# unique_train_prop = train_df["property"].unique()
# unique_train_prop = [("dummy_con", prop.strip(), int(0)) for prop in unique_train_prop]
# 
# unique_train_prop_df = pd.DataFrame.from_records(unique_train_prop)
# 
# unique_train_prop_df.to_csv("data/evaluation_data/nn_analysis/mcrae_unique_train_properties.tsv", sep="\t", header=None, index=None)
# 
# 
# print (len(unique_train_con))
# print (len(unique_train_prop))
# 
# 

# mc_test_file = "data/evaluation_data/extended_mcrae/test_mcrae.tsv"
# 
# test_df = pd.read_csv(mc_test_file, sep="\t", names=["concept", "property", "label"])
# 
# unique_test_con = test_df["concept"].unique()
# unique_test_con = [(con.strip(), "dummy_prop", int(0)) for con in unique_test_con]
# 
# unique_test_df = pd.DataFrame.from_records(unique_test_con)
# unique_test_df.to_csv("data/evaluation_data/nn_analysis/mcrae_unique_test_concepts.tsv", sep="\t", header=None, index=None)
# 
# 
# unique_test_prop = test_df["property"].unique()
# unique_test_prop = [("dummy_con", prop.strip(), int(0)) for prop in unique_test_prop]
# 
# unique_test_prop_df = pd.DataFrame.from_records(unique_test_prop)
# 
# unique_test_prop_df.to_csv("data/evaluation_data/nn_analysis/mcrae_unique_test_properties.tsv", sep="\t", header=None, index=None)
# 
# 
# print (len(unique_test_con))
# print (len(unique_test_prop))
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:


def transform(vecs):
    
    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []
    
    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm**2-np.linalg.norm(v)**2)))
    
    return new_vecs


# 

# In[ ]:


# Get the embeddings for property and concepts

def get_embedding (model, config):
    
    print (f"Config in get_embedding function : {config}")
    
    test_dataset, test_dataloader = mcrae_dataset_and_dataloader(
        dataset_params=config.get("dataset_params"),
        dataset_type="test",
        data_df=None,
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
            print (f"Concepts Data :", len(batch[0]))
            print (f"Concepts Data :", batch[0])
            print (f"concept_embedding.shape : {concept_embedding.shape}")
            
            print (f"Property Data :", len(batch[1]))
            print (f"Property Data :", batch[1])
            print (f"property_embedding.shape : {property_embedding.shape}")
            
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


#### Loading the BERT Large Model for generating Property Embedding
#### Here change the property test_file in config to the tsv file which contain the properties

# local_prop_config_file_path = "configs/nn_analysis/prop_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"
# hawk_bert_base_prop_config_file_path = "configs/nn_analysis/hawk_prop_nn_analysis_bert_base_fine_tune_mscg_adj_gkb_config.json"

hawk_bert_large_prop_config_file_path = "configs/nn_analysis/hawk_prop_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"



torch.cuda.empty_cache()

prop_config = read_config(hawk_bert_large_prop_config_file_path)
prop_model = load_pretrained_model(prop_config)
prop_model.eval()
prop_model.to(device)
print("Property Model Loaded")


# In[ ]:


_, _, prop_list, prop_emb = get_embedding(prop_model, prop_config)


# In[ ]:


print (f"prop_list len - {len(prop_list)}, Property Emb Len - {len(prop_emb)}")


# In[ ]:


prop_trans = transform(prop_emb)


# In[ ]:


prop_name_emb_dict = {"name_list_prop" : prop_list,
                      "untransformed_prop_emb":prop_emb,
                     "transformed_prop_emb" : prop_trans}


# In[ ]:


print (f"Pickling the transformed property name list and their embeddings.")

pickle_file_name = "/scratch/c.scmag3/biencoder_concept_property/data/evaluation_data/nn_analysis/music_hd/properties_music_hd_vocab_embeds.pkl"

with open (pickle_file_name, "wb") as f:
    pickle.dump(prop_name_emb_dict, f)
    


# In[ ]:


for key, value in prop_name_emb_dict.items():
    print (f"{key} : {len(value)}")

print ()
print ("*" * 50)
print (*prop_list, sep="\t")


# In[ ]:


print ("Fininshed Getting Property Embeddings....")
print ("Now Loading the model for Concepts Embeddings...")


# In[ ]:





# In[ ]:


# Loading the model model to generate concept embeddings
# Here change the concept test file the file where the test (query) concepts are loaded

torch.cuda.empty_cache()

# local_con_conf_file_path = "configs/nn_analysis/con_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"
# hawk_bert_base_con_config_file_path = "configs/nn_analysis/hawk_con_nn_analysis_bert_base_fine_tune_mscg_adj_gkb_config.json"

hawk_bert_large_con_conf_file_path = "configs/nn_analysis/hawk_con_nn_analysis_bert_large_fine_tune_mscg_adj_gkb_config.json"



con_config = read_config(hawk_bert_large_con_conf_file_path)
con_model = load_pretrained_model(con_config)
con_model.eval()
con_model.to(device)
print ("Concept Model Loaded")


# In[ ]:


con_list, con_emb, _, _ = get_embedding(con_model, con_config)


# In[ ]:


print (f"con_list len - {len(con_list)}, con_emb Len - {len(con_emb)}")


# In[ ]:


con_trans = transform(con_emb)


# In[ ]:


con_name_emb_dict = {"name_list_con" : con_list,
                     "untransformed_con_emb": con_emb,
                    "transformed_con_emb" : con_trans}


# In[ ]:


with open ("data/evaluation_data/nn_analysis/music_hd/concept_music_hd_test_embeds.pkl", "wb") as f:
    pickle.dump(con_name_emb_dict, f)


# In[ ]:


for key, value in con_name_emb_dict.items():
    print (f"{key} : {len(value)}")

print ()
print ("*" * 50)
print (*con_list, sep="\t")


# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# import pickle
# import numpy as np
# import nltk
# from sklearn.neighbors import NearestNeighbors
# from collections import Counter
# import pandas as pd
# from collections import Counter
# 

# 
# hd_con_emb_file = "/home/amitgajbhiye/cardiff_work/dot_product_model_nn_analysis/paper_concepts_name_emb.pickle"
# hd_prop_emb_file = "/home/amitgajbhiye/cardiff_work/dot_product_model_nn_analysis/paper_noun_properties.pkl"
# 
# with open(hd_con_emb_file, "rb") as con_emb, open(hd_prop_emb_file, "rb") as prop_emb:
#     
#     con_name_emb = pickle.load(con_emb)
#     prop_name_emb = pickle.load(prop_emb)
# 
# print (con_name_emb.keys())
# print (prop_name_emb.keys())

# print (f'Number of Properties in the loaded prop pickel : {len(prop_name_emb.get("name_list_prop"))}', flush=True)
# print (f'Number of Untransformed Properties Embedding in the loaded prop pickel : {len(prop_name_emb.get("untransformed_prop_emb"))}', flush=True)
# print (f'Number of TRansformed Properties Embedding in the loaded prop pickel : {len(prop_name_emb.get("transformed_prop_emb"))}', flush=True)
# 
# print ()
# print (f'Number of Concepts in the loaded con pickel : {len(con_name_emb.get("name_list_con"))}', flush=True)
# print (f'Number of Untransformed Concepts Embedding in the loaded prop pickel : {len(con_name_emb.get("untransformed_con_emb"))}', flush=True)
# print (f'Number of Transformed Concepts Embedding in the loaded prop pickel : {len(con_name_emb.get("transformed_con_emb"))}', flush=True)

# prop_name_emb.get("transformed_prop_emb")[0].shape

# prop_list = prop_name_emb.get("name_list_prop")
# del prop_name_emb.get("transformed_prop_emb")[(prop_list.index("fruit."))]

# # Learning Nearest Neighbours
# num_nearest_neighbours = 10
# nbrs = NearestNeighbors(n_neighbors=num_nearest_neighbours, algorithm='brute').fit(np.array(prop_name_emb.get("transformed_prop_emb")))

# distances, indices = nbrs.kneighbors(np.array(con_name_emb.get("transformed_con_emb")))

# print (indices)
# print (indices.shape)

# print (con_name_emb.keys())
# print (prop_name_emb.keys())

# len(prop_name_emb.get("untransformed_prop_emb"))

# for idx, con in zip(indices, con_name_emb.get("name_list_con")):    
#     print (f"{con} : {[prop_name_emb.get('name_list_prop') [prop_id] for prop_id in idx]}\n", flush=True)

# for idx, con in zip(indices, con_name_emb.get("name_list_con")):
#     
#     # print (f"{con} : {[prop_name_emb.get('name_list_prop') [prop_id] for prop_id in idx]}\n", flush=True)
#     
#     prop_list = [prop_name_emb.get('name_list_prop') [prop_id] for prop_id in idx]
#     
#     # prop_list = [prop.replace(".", "") for prop in prop_list]
#     
#     print (con , ":", prop_list)
#     
#     
#     print ()
#     

# d = {}
# for idx, con in zip(indices, con_name_emb.get("name_list_con")):
#     d[con] = [prop_name_emb.get('name_list_prop') [prop_id].strip() for prop_id in idx]

# 
# def pos_tagger(x):
#     
#     tokens = nltk.word_tokenize(x)
#     # print ("tokens :", tokens)
#     # print ("pos tags :", nltk.pos_tag(tokens))
#     return nltk.pos_tag(tokens)
#     
# 
# def filter_prop (con, prop_list):
#     
#     filtered_prop_list = []
#     
#     filtered_prop_ends_in_adj = []
#     filtered_hyp_ends_in_noun = []
#     
#     con = con.lower().strip()
#     prop_list = [prop.lower().strip() for prop in prop_list]
#     
#     for prop in prop_list:
#         if (con not in prop) and (prop not in con) :
#             # print (f"{con} : {prop}, {pos_tagger(prop)}, {pos_tagger(prop)[-1]}")
#             
#             if pos_tagger(prop)[-1][1] in ("NN","NNS","NNPS"):
#                 filtered_prop_list.append(prop)
#                 filtered_hyp_ends_in_noun.append(prop)
#             elif pos_tagger(prop)[-1][1] in ("JJ"):
#                 filtered_prop_ends_in_adj.append(prop)
#         
#             # print (f"filtered_prop_list : {filtered_prop_list}")
#     
#     print ("Property :", filtered_prop_ends_in_adj)
#     print ("Hypernym :", filtered_hyp_ends_in_noun)
#     print ("*"*20)
#     print ()
#     
#     
#     # print (len(filtered_prop_list))
#     # print (filtered_prop_list)
#     # print ()
#     
# #     filtered_prop_list = [prop.strip() for prop in filtered_prop_list]
# 
# #     if len(filtered_prop_list) >= 15:
# #         return filtered_prop_list[0:15]
# #     else:
# #         return filtered_prop_list
# 
#     
# 
# d = {}
# for idx, con in zip(indices, con_name_emb.get("name_list_con")):
#     
#     filtered_prop_list = []
#     
#     prop_for_con = [prop_name_emb.get('name_list_prop') [prop_id].strip() for prop_id in idx]
#     
#     print (f"concept : {con}")
#     # print (f"All properties : {prop_for_con}")
#     # print ([pos_tagger(prop) for prop in prop_for_con])
#     filtered_prop_list = filter_prop(con, prop_for_con)
#     
#     d[con] = filtered_prop_list
#     
# 

# 

# 

# l = []
# for key, value in d.items():
#     print (f"{key} : {len(value)}")
#     
#     l.append(len(value))
# 
# counts = Counter(l)
# 

# print (counts)

# print (len(d.keys()))

# df = pd.DataFrame.from_dict(d, orient="index")

# print (list(df.columns))

# df.reset_index(inplace=True, drop=False)

# df

# hypo_hyper_file_name = "/home/amitgajbhiye/cardiff_work/dot_product_model_nn_analysis/filtered_hd_test_results.csv"
# columns =["hypo","hyp=1","hyp=2","hyp=3","hyp=4","hyp=5","hyp=6","hyp=7","hyp=8","hyp=9","hyp=10","hyp=11","hyp=12","hyp=13","hyp=14","hyp=15"]
# 
# df.columns = columns
# 
# df["hypo"] = df["hypo"].str.strip()
# 
# df.to_csv(hypo_hyper_file_name, sep = ",", index=False, header=True)

# 

# 

# for idx, con in zip(indices, con_name_emb.get("name_list_con")):
#     print (f"{con} : {[prop_name_emb.get('name_list_prop') [prop_id] for prop_id in idx]}\n", flush=True)
# 

# 

# 

# In[ ]:





# In[ ]:




