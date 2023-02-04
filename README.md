## Dot Product Model For Concept Property Classification Task

## Model Details
The dot product model for concept property classification consists of two separate pre-trained Language Model (LM) based encoders. The concept encoder is trained on the input of the form `concept means [MASK]` and the property encoder on the form `property means [MASK]`. The vector encoding for the `[MASK]` is taken as the representation for the concept or the property. The dot product of the vector encodings of the concept and property is passed through the sigmoid activation to get the model prediction.

## Training Methodology 

### Pre-training on Different Data
The dot product model is first trained on the different types and amounts (100K and 500K) of data from the `Microsoft Concept Graph (mscg)`, `Generics KB Properties (gkb)` and `Prefix Adjectives`. The data can be found in the `data` directory of the repo. The model in this configuration uses in-batch negative sampling. The input file is a `tsv` in the form of `concept property`. The negatives are sampled via in-batch negative sampling during model training.  

The `neg_batch_sampling` branch of the codebase contains the latest code. 

Following are the steps to train the model:
- Clone the repo and checkout the `neg_batch_sampling` branch:
	- git clone git@github.com:amitgajbhiye/biencoder_concept_property.git
	- cd biencoder_concept_property/
	- git checkout neg_batch_sampling
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
- The models are also available on Hawk. I have changed the `/scratch` partition permission, so they are readable. The models are saved in : 
	- 100k Models - `/scratch/c.scmag3/biencoder_concept_property/trained_models/100k_data_experiments`
	- 500k Models - `/scratch/c.scmag3/biencoder_concept_property/trained_models/500k_trained_models`


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
