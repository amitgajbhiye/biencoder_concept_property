#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

import torch
import nltk
import pickle
import itertools

from model.concept_property_model import ConceptPropertyModel
from utils.functions import create_model
from utils.functions import load_pretrained_model
from utils.functions import read_config
from utils.functions import mcrae_dataset_and_dataloader
from utils.functions import compute_scores
from fine_tune import test_best_model

from sklearn.neighbors import NearestNeighbors
from collections import Counter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# assert os.environ["CONDA_DEFAULT_ENV"] == "gvenv", "Activate 'gvenv' conda environment"

print (f"Device Name : {device}")
print (f"Conda Environment Name : {os.environ['CONDA_DEFAULT_ENV']}")


import warnings
warnings.filterwarnings("ignore")


# mcrae_train_df = pd.read_csv("data/evaluation_data/extended_mcrae/train_mcrae.tsv", sep="\t", names=["concept", "property", "label"])
# mcrae_test_df = pd.read_csv("data/evaluation_data/extended_mcrae/test_mcrae.tsv", sep="\t", names=["concept", "property", "label"])
# 
# print ("McRae Train Df size : ", mcrae_train_df.shape)
# print (mcrae_train_df.head())
# 
# print ()
# 
# print ("McRae Test Df size : ", mcrae_test_df.shape)
# print (mcrae_test_df)
# 

# In[ ]:


def transform(vecs):
    
    maxnorm = max([np.linalg.norm(v) for v in vecs])
    new_vecs = []
    
    for v in vecs:
        new_vecs.append(np.insert(v, 0, np.sqrt(maxnorm**2-np.linalg.norm(v)**2)))
    
    return new_vecs


# In[ ]:



local_config_file_path = "configs/nn_analysis/nn_classifier_fine_tuned_bert_base_all_data.json"
hawk_config_file_path = "configs/nn_analysis/hawk_nn_classifier_fine_tuned_bert_base_all_data.json"

model_config = read_config(hawk_config_file_path)

print ("model_config")
print (model_config)

print ('model_config["dataset_params"]')
print (model_config["dataset_params"])


# In[ ]:


num_nearest_neighbours = 1


# In[ ]:


def predict_label(train_cons_similar_to_test, train_props_similar_to_test, train_df, test_df):

    preds = []

    for index, row in test_df.iterrows():
        
        print ()
        print ("Index :", index)
        test_con, test_prop, test_label = row["concept"], row["property"], row["label"]
        
        train_similar_props = train_props_similar_to_test.get(test_prop)
        
        assert train_similar_props is not None, "No Train Similar Properties for the Test Property"

        print ("Test Data :", index, test_con, test_prop, test_label)
        
        print ("Properties Similar to test property in Train File")
        print (train_similar_props)
        
        # print ("***************** Concept Processing Starts ***************** ")

        train_similar_concepts = train_cons_similar_to_test.get(test_con)
        
        assert train_similar_concepts is not None, "No Train Similar Concepts for the Test Concept"

        print ("Concepts Similar to test concept in Train File")
        print (train_similar_concepts)
        
        combination = list(itertools.product(train_similar_concepts, train_similar_props))
        
        print ("Combination of Similar Concept and Similar Property")
        print (combination)
        
        label_list = []
        
        print ("label_list")
        print (label_list)
        
        for con, prop in combination:
            df = train_df.loc[(train_df["concept"] == con) & (train_df["property"] == prop) & (train_df["label"] == 1)]
            print (f"Dataframe {con} : {prop} is Empty {df.empty}")
        
            if df.empty:
                label_list.append(0)
            else:
                label_list.append(1)
                
        print ("label_list")
        print (label_list)
        
        if num_nearest_neighbours == 3:
            threshold = 5
        elif num_nearest_neighbours == 1:
            threshold = 1
            
        label_sum = sum(label_list)
        
        print ("label_sum :", label_sum)
        print ("threshold :", threshold)
        
        if label_sum >= threshold:
            test_pred = 1
        else:
            test_pred = 0
        
        preds.append(test_pred)
    
    return preds


# In[ ]:


#### Loading the BERT Base Model for generating Property Embedding

torch.cuda.empty_cache()

local_base_path = "/home/amitgajbhiye/cardiff_work/dot_product_model_nn_analysis/mcrae_train_test_embeddings/con_prop_split_train_test_files"

hawk_base_path = "data/evaluation_data/nn_analysis/mcrae_con_prop_split_train_test_files"

all_gold_labels, all_preds  = [], []

for x in range(9):
    
    print()
    print (f"For Fold {x}")
    train_file_path = os.path.join(hawk_base_path, f"{x}_train_con_prop_split_con_prop.pkl")
    test_file_path = os.path.join(hawk_base_path, f"{x}_test_con_prop_split_con_prop.pkl")
    
    print (train_file_path)
    print (test_file_path)
    print ()
    
    with open (train_file_path, "rb") as train_file, open (test_file_path, "rb") as test_file:
        train_df = pickle.load(train_file)
        test_df = pickle.load(test_file)
    
    print (f"Train Df shape : {train_df.shape}, {train_df.columns}")
    print (f"Test Df shape : {test_df.shape}, {test_df.columns}")
    
    train_concept = train_df["concept"].unique()
    name_train_concept = train_concept
    
    train_prop = train_df["property"].unique()
    name_train_prop = train_prop
    
    
    train_concept = [(con, "dummy_prop", int(0)) for con in train_concept]
    train_prop = [("dummy_con", prop, int(0)) for prop in train_prop]
    
    test_concept = test_df["concept"].unique()
    name_test_concept = test_concept
    
    test_prop = test_df["property"].unique()
    name_test_prop = test_prop
    
    test_concept = [(con, "dummy_prop", int(0)) for con in test_concept]
    test_prop = [("dummy_con", prop, int(0)) for prop in test_prop]
    
    print (f"#Unique Train Concepts : {len(train_concept)}")
    print (f"#Unique Train Property : {len(train_prop)}")
    
    print ()
    print (f"#Unique Test Concepts : {len(test_concept)}")
    print (f"#Unique Test Property : {len(test_prop)}")
    
    print()
    print (f"Concept Intersection : {len(set(train_concept).intersection(test_concept))}")
    print (f"Property Intersection : {len(set(train_prop).intersection(test_prop))}")
    print ()
    
    for i, con_list in enumerate([train_concept, test_concept]):
        
        concept_df = pd.DataFrame.from_records(con_list, columns=["concept", "property", "label"])
        model_config["dataset_params"]["loader_params"]["batch_size"] = concept_df.shape[0]
        
        print()
        print (f"Concept i : {i}")
        print (concept_df.head())
        if i == 0:
            
            print ("Train Concepts")
            torch.cuda.empty_cache()
            
            print (f'Batch Size : {model_config["dataset_params"]["loader_params"]["batch_size"]}') 
            train_concept_embs, _, _, _ = test_best_model(model_config, test_df=concept_df, fold=None)
            print ("train_concept_embs.shape :", train_concept_embs.shape)
            
            train_concept_embs = [x.cpu().numpy() for x in train_concept_embs]
            transformed_train_concept_embs = transform(train_concept_embs)
            
            print (f"len(transformed_train_concept_embs) : {len(transformed_train_concept_embs)}")
            
        elif i == 1:
            print ("Test Concepts")
            torch.cuda.empty_cache()
            print (f'Batch Size : {model_config["dataset_params"]["loader_params"]["batch_size"]}') 
            test_concept_embs, _, _, _ = test_best_model(model_config, test_df=concept_df, fold=None)
            print ("test_concept_embs.shape :", test_concept_embs.shape)
            
            test_concept_embs = [x.cpu().numpy() for x in test_concept_embs]
               
            transformed_test_concept_embs = transform(test_concept_embs)
            
            print (f"len(transformed_test_concept_embs) : {len(transformed_test_concept_embs)}")
    
    for i, prop_list in enumerate([train_prop, test_prop]):
            
        property_df = pd.DataFrame.from_records(prop_list, columns=["concept", "property", "label"])       
        model_config["dataset_params"]["loader_params"]["batch_size"] = property_df.shape[0]
        
        print()
        print (f"Property i : {i}")
        print (property_df.head())
        
        if i == 0:
            
            print ("Train Property")
            torch.cuda.empty_cache()
            print (f'Batch Size : {model_config["dataset_params"]["loader_params"]["batch_size"]}')
            _, train_property_embs, _, _ = test_best_model(model_config, test_df=property_df, fold=None)
            print (f"train_property_embs.shape : {train_property_embs.shape}")
            
            train_property_embs = [x.cpu().numpy() for x in train_property_embs]
            transformed_train_property_embs = transform(train_property_embs)
            
            print (f"len(transformed_train_property_embs) : {len(transformed_train_property_embs)}")
        
        elif i ==1:
            
            print ("Test Property")
            torch.cuda.empty_cache()
            print (f'Batch Size : {model_config["dataset_params"]["loader_params"]["batch_size"]}')
            _, test_property_embs, _, _ = test_best_model(model_config, test_df=property_df, fold=None)
            print (f"test_property_embs.shape : {test_property_embs.shape}")
            
            test_property_embs = [x.cpu().numpy() for x in test_property_embs]
            transformed_test_property_embs = transform(test_property_embs)
            
            print (f"len(transformed_test_property_embs) : {len(transformed_test_property_embs)}")
            
    # print (" **************************** Concept Nearest Neighbours ****************************")
    
    train_con_nbrs = NearestNeighbors(n_neighbors=num_nearest_neighbours, algorithm='brute', metric='euclidean').fit(np.array(transformed_train_concept_embs))
    con_test_distances, con_test_indices = train_con_nbrs.kneighbors(np.array(transformed_test_concept_embs))
    
    train_cons_similar_to_test = {}
    
    for idx, con in zip(con_test_indices, name_test_concept):

        train_cons_similar_to_test[con] = [name_train_concept[con_id] for con_id in idx]
        # print (f"{con} : {train_cons_similar_to_test[con]}")
    
    # print (" **************************** Property Nearest Neighbours ****************************")
    
    train_prop_nbrs = NearestNeighbors(n_neighbors=num_nearest_neighbours, algorithm='brute', metric='euclidean').fit(np.array(transformed_train_property_embs))
    prop_test_distances, prop_test_indices = train_prop_nbrs.kneighbors(np.array(transformed_test_property_embs))
    
    train_props_similar_to_test = {}
    
    for idx, prop in zip(prop_test_indices, name_test_prop):
        
        train_props_similar_to_test[prop] = [name_train_prop[prop_id] for prop_id in idx]
        # print (f"{prop} : {train_props_similar_to_test[prop]}")
    
    
    gold_label_for_fold = test_df["label"].values
    pred_for_fold = predict_label(train_cons_similar_to_test, train_props_similar_to_test, train_df, test_df)
    
    all_gold_labels.extend(gold_label_for_fold)
    all_preds.extend(pred_for_fold)
    
    
all_gold_labels = np.array(all_gold_labels)
all_preds = np.array(all_preds)

print ("Finished")


# In[ ]:


len(all_preds)


# In[ ]:


len(all_gold_labels)


# In[ ]:


assert len(all_gold_labels) == len(all_preds)


# In[ ]:


print (Counter(all_preds))
print (Counter(all_gold_labels))


# In[ ]:


results = compute_scores(all_gold_labels, all_preds)


# In[ ]:


print ()
print ("Concept Property Split")
print (f"NN Classifier with pretrained BERT Base Embedding pretrained on MSCG+PREFIX+GKB Data")
print (f"Nearest Neighbours Considered : {num_nearest_neighbours}")
print ()

for key, value in results.items():
    print (key, value)


# In[ ]:





# In[ ]:




