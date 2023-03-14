#!/bin/bash --login 

mkdir -p logs 
mkdir -p trained_models/embeddings

cd trained_models

# bert-large-uncased trained on conceptnet premium
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bienc_bert_large_cnetp_pretrain_lr26.pt

# bert-base-uncased trained on conceptnet premium
wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_gkb_cnet_plus_cnet_has_property.pt

# bert-large-uncased trained on conceptnet premium
# wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_mscg_prefix_adjective_gkb.pt