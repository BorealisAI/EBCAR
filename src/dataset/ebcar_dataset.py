# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class EBCARDatasetReal(Dataset):
    """
    Dataset for the EBCAR model on real datasets.
    """

    def __init__(
        self,
        query_list: torch.Tensor,
        passage_list: torch.Tensor,
        label_list: torch.Tensor,
        document_id_list: list,
        passage_id_list: list,
        passage_text_list: list,
        query_text_list: list,
        cfg: DictConfig,
        size: int = None,
    ):
        self.query_list = query_list
        self.passage_list = passage_list
        self.label_list = label_list
        self.document_id_list = document_id_list
        self.passage_id_list = passage_id_list
        self.passage_text_list = passage_text_list
        self.query_text_list = query_text_list

        # Randomly shuffle the data, since we are using a mixture of three datasets for training
        indices = torch.randperm(len(self.query_list))
        self.query_list = self.query_list[indices]
        self.passage_list = self.passage_list[indices]
        self.label_list = self.label_list[indices]
        self.document_id_list = [self.document_id_list[i] for i in indices]
        self.passage_id_list = [self.passage_id_list[i] for i in indices]
        self.passage_text_list = [self.passage_text_list[i] for i in indices]
        self.query_text_list = [self.query_text_list[i] for i in indices]

        self.cfg = cfg
        if size is not None:
            self.query_list = self.query_list[:size]
            self.passage_list = self.passage_list[:size]
            self.label_list = self.label_list[:size]
            self.document_id_list = self.document_id_list[:size]
            self.passage_id_list = self.passage_id_list[:size]
            self.passage_text_list = self.passage_text_list[:size]
            self.query_text_list = self.query_text_list[:size]

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        # Change the absolute document id to relative document id
        temp_document_id_list = self.document_id_list[idx]
        temp_passage_id_list = self.passage_id_list[idx]
        temp_passage_id_list = [int(passage_id) for passage_id in temp_passage_id_list]
        unique_document_ids = []
        for doc_id in temp_document_id_list:
            if doc_id not in unique_document_ids:
                unique_document_ids.append(doc_id)
        temp_document_id_list = [
            unique_document_ids.index(doc_id) for doc_id in temp_document_id_list
        ]
        self.document_id_list[idx] = torch.tensor(temp_document_id_list)
        self.passage_id_list[idx] = torch.tensor(temp_passage_id_list)
        # If we add the positional encoding for the document id and passage id, we need to randomly permute the order of retrieved passages every time
        # Otherwise the efficacy of the retriever will matter here, and the model will learn to rely on the retriever's output
        if self.cfg.add_positional_encoding:
            indices = torch.randperm(self.cfg.retrieval.top_k)
            query = self.query_list[idx]
            passage = self.passage_list[idx][indices]
            label = self.label_list[idx][indices]
            document_id = self.document_id_list[idx][indices]
            passage_id = self.passage_id_list[idx][indices]
            passage_text = [self.passage_text_list[idx][i] for i in indices]
            query_text = self.query_text_list[idx]
        else:
            query = self.query_list[idx]
            passage = self.passage_list[idx]
            label = self.label_list[idx]
            document_id = self.document_id_list[idx]
            passage_id = self.passage_id_list[idx]
            passage_text = self.passage_text_list[idx]
            query_text = self.query_text_list[idx]

        return {
            "query": query,
            "passage": passage,
            "label": label,
            "document_id": document_id,
            "passage_id": passage_id,
            "passage_text": passage_text,
            "query_text": query_text,
        }

    def collate_fn(self, batch):
        return {
            "query": torch.stack([item["query"] for item in batch]),
            "passage": torch.stack([item["passage"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "document_id": torch.stack([item["document_id"] for item in batch]),
            "passage_id": torch.stack([item["passage_id"] for item in batch]),
            "passage_text": [item["passage_text"] for item in batch],
            "query_text": [item["query_text"] for item in batch],
        }
