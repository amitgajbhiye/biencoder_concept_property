#!/bin/bash --login 

mkdir -p logs 
mkdir -p trained_models/embeddings

cd trained_models


echo "Downloading Pretrained Models ..."

# bert-large-uncased trained on conceptnet premium
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bienc_bert_large_cnetp_pretrain_lr26.pt

# bert-base-uncased trained on conceptnet premium
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_gkb_cnet_plus_cnet_has_property.pt

# bert-large-uncased trained on conceptnet premium
# wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_mscg_prefix_adjective_gkb.pt


# Date 5th May 2023

# Contrastive and Cross Entropy Jointly Trained Model
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/entropy_infonce_joint_loss_cnetp_pretrain_bb_bienc_bert_base_uncased.pt

# Contrastive Biencoders

#Contrastive Concept Property fix Model
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/conprop_fix_infonce_cnetp_pretrain_bb_bienc_bert_base_uncased.pt

#Contrastive Concept fix Model
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/contastive_bienc_cnetp_pretrain_bert_base_uncased.pt

#Contrastive Property fix Model
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/prop_fix_bienc_infonce_bert_base_cnetp_pretrain.pt

