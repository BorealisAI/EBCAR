# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os
import time

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.dataset import EBCARDatasetReal
from src.model.ebcar_dedicated_attention_model import EBCarRerankerHybridAttention
from src.utils import calculate_MRR, count_trainable_parameters


def train_ebcar(cfg: DictConfig):
    save_dir = os.path.join(cfg.save_dir, cfg.dataset.name)

    if cfg.use_cuda:
        device = torch.device(
            f"cuda:{cfg.n_cuda}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    query_list = torch.load(os.path.join(save_dir, "query_list.pt"))
    query_list = query_list.float().cpu()
    passage_list = torch.load(os.path.join(save_dir, "passage_list.pt"))
    passage_list = passage_list.float().cpu()
    label_list = torch.load(os.path.join(save_dir, "label_list.pt"))
    label_list = label_list.float().cpu()
    document_id_list = json.load(open(os.path.join(save_dir, "document_id_list.json")))
    passage_id_list = json.load(open(os.path.join(save_dir, "passage_id_list.json")))
    passage_text_list = json.load(
        open(os.path.join(save_dir, "passage_text_list.json"))
    )
    query_text_list = json.load(open(os.path.join(save_dir, "query_text_list.json")))
    print(query_list.shape)
    print(passage_list.shape)
    print(label_list.shape)
    print(document_id_list[0])
    print(passage_id_list[0])
    print(passage_text_list[0])
    print(query_text_list[0])

    if cfg.validation_dataset is not None:
        val_dir = os.path.join(cfg.save_dir, cfg.validation_dataset.name)
        val_query_list = torch.load(os.path.join(val_dir, "query_list.pt"))
        val_passage_list = torch.load(os.path.join(val_dir, "passage_list.pt"))
        val_label_list = torch.load(os.path.join(val_dir, "label_list.pt"))
        val_document_id_list = json.load(
            open(os.path.join(val_dir, "document_id_list.json"))
        )
        val_passage_id_list = json.load(
            open(os.path.join(val_dir, "passage_id_list.json"))
        )
        val_passage_text_list = json.load(
            open(os.path.join(val_dir, "passage_text_list.json"))
        )
        val_query_text_list = json.load(
            open(os.path.join(val_dir, "query_text_list.json"))
        )

    dataset = EBCARDatasetReal(
        query_list,
        passage_list,
        label_list,
        document_id_list,
        passage_id_list,
        passage_text_list,
        query_text_list,
        cfg,
        size=cfg.training_data_size,
    )
    val_dataset = EBCARDatasetReal(
        val_query_list,
        val_passage_list,
        val_label_list,
        val_document_id_list,
        val_passage_id_list,
        val_passage_text_list,
        val_query_text_list,
        cfg,
        size=cfg.validation_data_size,
    )
    if cfg.validation_dataset is not None:
        train_dataset = dataset
        val_dataset = val_dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [1 - cfg.validation_data_ratio, cfg.validation_data_ratio]
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    if cfg.add_positional_encoding:
        run_name = f"pos_emb_ebcar_lr{cfg.learning_rate}_wd{cfg.weight_decay}"
    else:
        run_name = f"vanilla_ebcar_lr{cfg.learning_rate}_wd{cfg.weight_decay}"

    if cfg.use_dedicated_attention:
        run_name = "hybrid_attention_" + run_name

    run_name = cfg.dataset.name + "_" + run_name

    run_name += f"_top_k_{cfg.retrieval.top_k}"

    if cfg.log_wandb:
        logger = wandb.init(
            # Set the wandb entity where your project will be logged.
            entity="{wandb_entity}",
            # Set the wandb project where this run will be logged.
            project="EBCAR",
            # Track Config.
            config=OmegaConf.to_container(cfg, resolve=True),
            # Set the name of the run
            name=run_name,
        )

    # Initialize the reranker and optimizer
    reranker = EBCarRerankerHybridAttention(cfg, device)
    optimizer = torch.optim.Adam(
        reranker.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = None
    best_val_loss = float("inf")
    no_improvement_epochs = 0
    training_step = 0

    # Print the number of trainable parameters
    reranker.train()
    print(
        f"Number of trainable parameters in EBCAR: {count_trainable_parameters(reranker)}"
    )

    # Record the time used to inference the entire validation set
    total_val_time = []

    # Training and validation loss before training
    train_loss = 0
    reranker.eval()
    with torch.no_grad():
        for batch in train_loader:
            query = batch["query"].to(device)
            passage = batch["passage"].to(device)
            label = batch["label"].to(device)
            document_id = batch["document_id"].to(device)
            passage_id = batch["passage_id"].to(device)
            passage_text = batch["passage_text"]
            loss = reranker(
                query, passage, label, document_id, passage_id, passage_text
            )
            train_loss += loss.item()
        print(f"Training loss before training: {train_loss / len(train_loader)}")
    val_loss = 0
    reranker.eval()
    start_time = time.time()
    with torch.no_grad():
        reranked_all_passages = []  # [num_val_samples, num_passages]
        ground_truth_all_passages = []  # [num_val_samples]
        for batch in val_loader:
            query = batch["query"].to(device)
            passage = batch["passage"].to(device)
            label = batch["label"].to(device)
            document_id = batch["document_id"].to(device)
            passage_id = batch["passage_id"].to(device)
            passage_text = batch["passage_text"]
            loss = reranker(
                query, passage, label, document_id, passage_id, passage_text
            )
            val_loss += loss.item()
            reranked_passages, _ = reranker.rerank(
                query, passage, document_id, passage_id, passage_text
            )  # [batch_size, num_passages]
            reranked_all_passages.extend(reranked_passages)
            ground_truth_all_passages.extend(
                [passage_text[i][label[i].argmax()] for i in range(len(passage_text))]
            )  # [num_val_samples]
        end_time = time.time()
        total_val_time.append(end_time - start_time)
        val_mrr = calculate_MRR(
            reranked_all_passages, ground_truth_all_passages, top_k=cfg.MRR_at
        )
        print(f"Validation loss before training: {val_loss / len(val_loader)}")
        print(f"Validation MRR@{cfg.MRR_at}: {val_mrr}")
    if cfg.log_wandb:
        logger.log(
            {
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "epoch": 0,
                f"val_mrr@{cfg.MRR_at}": val_mrr,
            }
        )

    # Training loop
    for epoch in range(cfg.num_epochs):
        train_loss = 0
        reranker.train()
        for batch in train_loader:
            query = batch["query"].to(device)  # [batch_size, hidden_size]
            passage = batch["passage"].to(
                device
            )  # [batch_size, num_passages, hidden_size]
            label = batch["label"].to(device)  # [batch_size, num_passages]
            document_id = batch["document_id"].to(device)  # [batch_size, num_passages]
            passage_id = batch["passage_id"].to(device)  # [batch_size, num_passages]
            passage_text = batch["passage_text"]  # [batch_size, num_passages]
            loss = reranker(
                query, passage, label, document_id, passage_id, passage_text
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss += loss.item()
            if cfg.log_wandb:
                logger.log(
                    {"training_step": training_step, "training_step_loss": loss.item()}
                )
            training_step += 1

        print(
            f"Epoch {epoch + 1} training loss: {train_loss / len(train_loader)}",
        )

        val_loss = 0
        reranker.eval()
        start_time = time.time()
        with torch.no_grad():
            reranked_all_passages = []  # [num_val_samples, num_passages]
            ground_truth_all_passages = []  # [num_val_samples]
            for batch in val_loader:
                query = batch["query"].to(device)
                passage = batch["passage"].to(device)
                label = batch["label"].to(device)
                document_id = batch["document_id"].to(device)
                passage_id = batch["passage_id"].to(device)
                passage_text = batch["passage_text"]
                loss = reranker(
                    query, passage, label, document_id, passage_id, passage_text
                )
                val_loss += loss.item()
                reranked_passages, _ = reranker.rerank(
                    query, passage, document_id, passage_id, passage_text
                )  # [batch_size, num_passages]
                reranked_all_passages.extend(reranked_passages)
                ground_truth_all_passages.extend(
                    [
                        passage_text[i][label[i].argmax()]
                        for i in range(len(passage_text))
                    ]
                )  # [num_val_samples]
        end_time = time.time()
        total_val_time.append(end_time - start_time)
        val_mrr = calculate_MRR(
            reranked_all_passages, ground_truth_all_passages, top_k=cfg.MRR_at
        )
        print(
            f"Epoch {epoch + 1} validation loss: {val_loss / len(val_loader)}", end=", "
        )
        print(f"validation MRR@{cfg.MRR_at}: {val_mrr}")

        if cfg.log_wandb:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / len(train_loader),
                    "val_loss": val_loss / len(val_loader),
                    f"val_mrr@{cfg.MRR_at}": val_mrr,
                }
            )

        if val_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            torch.save(
                reranker.state_dict(),
                os.path.join(
                    save_dir,
                    f"{run_name}_{epoch + 1}.pt",
                ),
            )
            print(
                f"Saved model to {os.path.join(save_dir, f'{run_name}_{epoch + 1}.pt')}"
            )
            print(f"Best validation loss: {best_val_loss}")
            print(f"Best validation epoch: {epoch + 1}")
            no_improvement_epochs = 0  # Reset the counter
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= cfg.patience:
                print(f"No improvement for {cfg.patience} epochs, stopping training")
                break

    if cfg.log_wandb:
        logger.finish()

    print(
        f"Average time used to inference the entire validation set: {sum(total_val_time) / len(total_val_time)} seconds"
    )

    return reranker
