# Embedding-Based-Context-Aware-Reranker

This repository contains the official implementation for the paper ["Embedding-Based Context-Aware Reranker (EBCAR)"](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9dae2a90bae49dc874ce1ca8fcc20879-Abstract-Conference.html), which is accepted by ICLR 2026.

## Folder Structure
```bash
|- conf/  # Configuration files
    |- dataset/  # Dataset configuration
    |- model/  # Model configuration
    |- __init__.py  # Empty file to make the directory a package
    |- config.yaml  # Global configuration
|- data/  # Data files
    |- real/  # Preprocess and store the date of ConTEB train, ConTEB test, and MS_MARCO_v2
|- saved/  # Save cached data and model checkpoints
|- src/  # Source code for the project
|- .gitignore  # Git ignore file
|- main.py  # Main file to run the project
|- pyproject.toml  # Config isort and black to better format the code
|- README.md  # This file
|- requirements.txt  # necessary dependencies for pip install
|- utils.py  # Utility functions for the entire project
```

## Install Dependencies
We can automaticaly create a conda environment through:
```bash
conda create -n ebcar python=3.12.9
conda activate ebcar
pip install -r requirements.txt
```

All experiments are run on a single A100 GPU. It will take around 30 hours to train the EBCAR model.

## Run Experiments
Change all `{working_dir}` to the directory of cloned this repo on your machine. Then follow instructions below step by step.

### Preprocess Necessary Data
```bash
# Prepare the data for the main experiment, will automatically download and preprocess it
python3 {working_dir}/data/real/ConTEB_train/preprocess.py
python3 {working_dir}/data/real/ConTEB_test/preprocess.py
```

### Build Vector Database for Each Dataset for Retrieval
```bash
# Build VectorDatabase for Training
python3 main.py mode=build_vector_database dataset=conteb_train

# Build VectorDatabase for Validation
python3 main.py mode=build_vector_database dataset=conteb_val

# Build VectorDatabase for Test
python3 main.py mode=build_vector_database dataset=conteb_test_01MLDR
python3 main.py mode=build_vector_database dataset=conteb_test_02SQuAD
python3 main.py mode=build_vector_database dataset=conteb_test_03NarrativeQA
python3 main.py mode=build_vector_database dataset=conteb_test_04COVID_QA
python3 main.py mode=build_vector_database dataset=conteb_test_05ESG_Reports
python3 main.py mode=build_vector_database dataset=conteb_test_06Football
python3 main.py mode=build_vector_database dataset=conteb_test_07Geography
python3 main.py mode=build_vector_database dataset=conteb_test_08Insurance
```

### Cache the Training and Test Data to Embeddings
```bash
# Cache Training Set
python3 main.py mode=load_conteb_dataset dataset=conteb_train
python3 saved/conteb_train/merge.py

# Cache Validation Set
python3 main.py mode=load_conteb_dataset dataset=conteb_val

# Cache Test Set
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_01MLDR
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_02SQuAD
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_03NarrativeQA
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_04COVID_QA
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_05ESG_Reports
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_06Football
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_07Geography
python3 main.py mode=load_conteb_test_dataset dataset=conteb_test_08Insurance
```

### Train EBCAR Model
```bash
# Will log into WandB by setting {wandb_entity} up accordingly
python3 main.py mode=train_ebcar
```

### Evaluate
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # For deterministic algorithms

test_dataset_names=('conteb_test_01MLDR' 'conteb_test_02SQuAD' 'conteb_test_03NarrativeQA' 'conteb_test_04COVID_QA' 'conteb_test_05ESG_Reports' 'conteb_test_06Football' 'conteb_test_07Geography' 'conteb_test_08Insurance')

for test_dataset_name in ${test_dataset_names[@]}; do
    python3 main.py mode=evaluate_EBCAR test_dataset_name=${test_dataset_name}
done
```


## Citation
If you find EBCAR useful in your research, please consider citing:
```bibtex
@inproceedings{yuan2026embeddingbased,
    title={Embedding-Based Context-Aware Reranker},
    author={Ye Yuan and Mohammad Amin Shabani and Siqi Liu},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=OBMcxeSK5U}
}
```
