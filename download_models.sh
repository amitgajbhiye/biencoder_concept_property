#!/bin/bash --login 

mkdir logs 
mkdir -p trained_models/embeddings

cd trained_models

wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_gkb_cnet_plus_cnet_has_property.pt
# wget https://huggingface.co/amitgajbhiye/biencoder_concept_property_pretrained_models/resolve/main/bb_mscg_prefix_adjective_gkb.pt