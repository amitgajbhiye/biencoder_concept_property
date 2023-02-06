## BiEncoder Model For Concept Property Classification

[Modelling Commonsense Properties using Pre-Trained Bi-Encoders](https://aclanthology.org/2022.coling-1.349/)

## Model Details
The BiEncoder model for concept property classification consists of two separate pre-trained Language Model (LM) based encoders. The concept encoder is trained on the prompt `concept means [MASK]` and the property encoder on `property means [MASK]`. The vector encoding for the `[MASK]` is taken as the representation for the concept or the property. The dot product of the `[MASK]` embeddings of the concept and property is passed through the sigmoid activation to get the model prediction.


## Getting Concept and Property Embedding from Pretrained Models

The BiEncoder model can generate the embeddings of concept and properties. Please follow the following steps to get the embeddings. 

- Run the `download_models.sh` bash script. This will download two `BERT-base-uncased` pretrained models. First the model pretrained on ConceptNet data in Generics KB plus the `has_property` relation data in the Concept Net. We refer to this data as `conceptnet_premium`. The second model is pretrained on `Microsoft Concept Graph (mscg)`, `Generics KB Properties (gkb)` and `Prefix Adjectives`. The model will be downloaded in the `trained_models` directory.
    - `bash download_models.sh`

- Once the model is downloaded run the `get_embedding.py` module with the configuration file as follows:

	- `python3 get_embedding.py --config_file configs/generate_embeddings/get_concept_property_embeddings.json`

- The above script will generate a pickled dictionary of concept/property in the `save_dir` field of configuration file.

- Following are the important fields of the configuration file:
	- `dataset_name` - Name that will be used to save the embedding pickle file at the directory path specified in `save_dir` field.
	- `hf_checkpoint_name` and `hf_tokenizer_name` - The huggingface pretrained model ID and tokenizer name. For example, `bert-base-uncased`.
	- `context_num` - Context ID used in pretraining the models. We used context ID - 6.
	- `pretrained_model_path` - Path of the pretrained model. It is `trained_models/bb_gkb_cnet_plus_cnet_has_property.pt` or `trained_models/bb_mscg_prefix_adjective_gkb.pt` for the two downloaded models.
	- `get_con_prop_embeds` - Flag set to `true` to get concept or property embeddings.
	- `input_file_name`: Path of the input concept, property or concept and property file. 
	- `input_data_type` - Type of the embddings to generate. 
		- `concept` : for concept embeddings. The input file must be a file with each concept in one line.
		- `property` : for property embeddings. The input file must be a file with each property in one line.
		- `concept_and_property` : for the concept and property embeddings. The input file must file with each concept and associated property, one per line, separated by tab. 
	

## Training Methodology 

### Pre-training on Different Data
The biencoder model is first trained on the different types and amounts (100K and 500K) of data from the `Microsoft Concept Graph (mscg)`, `Generics KB Properties (gkb)` and `Prefix Adjectives`. The data can be found in the `data` directory of the repo. The model in this configuration uses in-batch negative sampling. The input file is a `tsv` in the form of `concept property`. The negatives are sampled via in-batch negative sampling during model training.  

<!-- The `neg_batch_sampling` branch of the codebase contains the latest code.  -->

Following are the steps to train the model:
- Clone the repo and checkout the `neg_batch_sampling` branch:
	- git clone git@github.com:amitgajbhiye/biencoder_concept_property.git
	- cd biencoder_concept_property/
	<!-- - git checkout neg_batch_sampling -->
- Create `logs` and `trained_models` directories:
	- mkdir logs trained_models 
- The model is trained with a configuration file that contains all the parameters for the datasets, model and training. 
-  The log file for the experiments are created in the `logs` directory. This can be changed in the `set_logger`
function in the `utils/functions.py` module. The name of the log file is of the form`log_experiment_name_timestamp`. The `experiment_name` comes from config file and timestamp is current timestamp.
-  `trained_models` is the directory where the trained model wil be saved. This can be changed in the `export_path` parameter of the config file.
-  In the config file, change the `hf_tokenizer_path` and `hf_model_path` to the paths of the downloaded tokenizer and pretrained language model. 
- To train the model execute the `run_model.py` script with the config file path as an argument.
- For example, to train the model on the 100K mscg data. Run the following command: 
	 - `python run_model.py --config_file configs/sample_configs/top_100k_mscg_config.json`
- The configuration files I used for the experiments are in self-descriptive directory names in the `configs` directory. The names of the config files and data files are also self-descriptive.
- The best-trained model is saved at the path specified in the `export_path`  with the name specified in the `model_name` parameter of the configuration file.

- The models trained on 100k and 500k different datasets are saved in One Drive at the [link](https://cf.sharepoint.com/:f:/t/ELEXIR/EvB5Kj7yY_pLp8uExM6xqVYBl2PIz-uMsGBMwICmR8Se_A?e=bnnoN7 )

<!-- - The models are also available on Hawk. I have changed the `/scratch` partition permission, so they are readable. The models are saved in : 
	- 100k Models - `/scratch/c.scmag3/biencoder_concept_property/trained_models/100k_data_experiments`
	- 500k Models - `/scratch/c.scmag3/biencoder_concept_property/trained_models/500k_trained_models` -->


### Fine-tuning Trained Model

The models trained above are fine tuned on the on the extended McRae dataset. The processed train file is `data/evaluation_data/extended_mcrae/train_mcrae.tsv` and the test file is `data/evaluation_data/extended_mcrae/test_mcrae.tsv`.

On the McRae data, the model is fine-tuned in three splits of the whole data:
- Default - Concept Split
- Property Split
- Concept Property Split

In the `Property` and `Concept Property` split settings the model uses cross-validation. 

Following are the steps to fine-tune a pretrained model:

- To fine tune the trained model use the sample configuration file - `configs/sample_configs/cv_sample_config_file.json`
- In the config file, specify the path of the following parameters: 
	- `pretrained_model_path` - The path of the pre-trained model that need to be fine-tuned (taken from the 100k and 500k trained model path specified above).
	- `do_cv` - `true` for property split cross validation and concept property split cross validation. `do_cv` is `false` for finetuning on default concept split.   
	- `cv_type` from `model_evaluation_property_split` and `model_evaluation_concept_property_split`
- To fine tuning the model execute the `fine_tune.py` script with the config file path as argument.
- For example, to fine the model trained on `100k mscg` data with `Property` split run the following command:
	- `python3 fine_tune.py --config_file configs/sample_configs/pcv_sample_config_file.json`



## Citation

```

@inproceedings{gajbhiye2022modelling,
    title = "Modelling Commonsense Properties Using Pre-Trained Bi-Encoders",
    author = "Gajbhiye, Amit  and
      Espinosa-Anke, Luis  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.349",
    pages = "3971--3983"
}

```