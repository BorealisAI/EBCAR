# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
The main file to run the experiments
"""

import os

import hydra
import torch
from omegaconf import OmegaConf

from src.build_vector_database import build_vector_database_from_corpus
from src.dataset import load_conteb_dataset, load_conteb_test_dataset
from src.evaluate import evaluate_EBCAR
from src.train_ebcar import train_ebcar
from utils import set_seed


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    """
    The main function to run the experiments
    cfg: the configuration
    """
    set_seed(cfg.seed)
    if "evaluate" in cfg.mode:
        import warnings

        warnings.filterwarnings("ignore")
        torch.use_deterministic_algorithms(True)
    os.chdir(cfg.working_dir)

    if cfg.mode == "build_vector_database":
        vectorstore = build_vector_database_from_corpus(cfg)
    elif cfg.mode == "load_conteb_dataset":
        load_conteb_dataset(cfg)
    elif cfg.mode == "load_conteb_test_dataset":
        load_conteb_test_dataset(cfg)
    elif cfg.mode == "train_ebcar":
        train_ebcar(cfg)
    elif cfg.mode == "evaluate_EBCAR":
        evaluate_EBCAR(cfg)
    else:
        raise NotImplementedError(f"Mode {cfg.mode} is not implemented.")


if __name__ == "__main__":
    main()
