# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.model.transformer_encoder_hybrid_attention import (
    TransformerEncoderWithHybridAttention,
)


class EBCarRerankerHybridAttention(nn.Module):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super(EBCarRerankerHybridAttention, self).__init__()
        self.cfg = cfg
        self.model = TransformerEncoderWithHybridAttention(
            num_layers=cfg.num_layers,
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
        )
        self.model.float()
        self.model.to(device)
        self.device = device
        self.num_heads = cfg.nhead

        # Initialize the positional encoding for document ids, we don't need to update the document id embedding
        self.document_id_embedding = nn.Embedding(
            num_embeddings=cfg.retrieval.top_k, embedding_dim=cfg.d_model
        )  # [top_k, d_model]
        self.document_id_embedding = self.document_id_embedding.to(device)
        self.document_id_embedding = self.document_id_embedding.float()
        self.document_id_embedding.requires_grad = False
        self.document_id_embedding.weight.data.normal_(0, 0.02)

        # Initialize the positional encoding for passage ids, we don't need to update the passage id embedding
        # Conteb on average has 23.5 passages per document, so we use 5000 as the maximum number of passages should be far enough
        passage_id_embedding = self.get_passage_positional_encoding(
            torch.tensor([list(range(5000))], device=device)
        ).squeeze(
            0
        )  # [5000, d_model]
        self.passage_id_embedding = nn.Embedding.from_pretrained(
            passage_id_embedding, freeze=True
        )  # [top_k, d_model]
        self.passage_id_embedding = self.passage_id_embedding.to(device)
        self.passage_id_embedding = self.passage_id_embedding.float()
        self.passage_id_embedding.requires_grad = False

    def get_passage_positional_encoding(self, input_ids: torch.Tensor):
        """
        Using sinusoidal positional encoding to encode the passage's position in the original document
        input_ids: [batch_size, max_length]
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices [seq_len]
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.float32)

        # Create dimension indices for even positions [d_model//2]
        div_term = torch.exp(
            torch.arange(
                0, self.cfg.d_model, 2, dtype=torch.float32, device=input_ids.device
            )
            * -(math.log(10000.0) / self.cfg.d_model)
        )

        # Initialize positional encoding tensor
        positional_encoding = torch.zeros(
            batch_size, seq_len, self.cfg.d_model, device=input_ids.device
        )

        # Reshape for proper broadcasting: positions [seq_len, 1] * div_term [1, d_model//2]
        positions = positions.unsqueeze(1)  # [seq_len, 1]
        div_term = div_term.unsqueeze(0)  # [1, d_model//2]

        # Calculate sin and cos values [seq_len, d_model//2]
        sin_values = torch.sin(positions * div_term)
        cos_values = torch.cos(positions * div_term)

        # Assign to even and odd positions
        positional_encoding[:, :, 0::2] = sin_values  # Even dimensions
        positional_encoding[:, :, 1::2] = cos_values  # Odd dimensions

        # Normalize the passage id embedding
        positional_encoding = torch.nn.functional.normalize(
            positional_encoding, p=2, dim=2
        )  # [batch_size, seq_len, d_model]

        return positional_encoding

    def forward(
        self,
        query: torch.Tensor,
        passages: torch.Tensor,
        labels: torch.Tensor,
        document_id: torch.Tensor,
        passage_id: torch.Tensor,
        passage_text: list[list[str]],
    ):
        """
        query: embeddings of the query [batch_size, hidden_size]
        passages: embeddings of the passages [batch_size, num_passages, hidden_size]
        labels: labels of the passages [batch_size, num_passages]
        document_id: document ids of the passages [batch_size, num_passages]
        passage_id: passage ids of the passages [batch_size, num_passages]
        passage_text: text of the passages [batch_size, num_passages]
        """

        # To model the dependency between passages, we can use a transformer encoder to model the interaction between passages
        # We can concatenate the embeddings of the passages and the query, and then pass it through the transformer encoder
        # The output of the transformer encoder will be the embeddings of the passages and the query
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Add positional encoding for document ids
        if self.cfg.add_positional_encoding:
            # document_id might be out of range, so we need to modulate it and plus the quotient
            document_id_embeddings = self.document_id_embedding(
                (document_id.long() % self.cfg.retrieval.top_k)
                + (document_id.long() // self.cfg.retrieval.top_k)
            )  # [batch_size, num_passages, hidden_size]
            passage_id_embeddings = self.passage_id_embedding(
                passage_id.long()
            )  # [batch_size, num_passages, hidden_size]
            # Directly add the positional encoding to the passages
            passages = (
                passages + document_id_embeddings + passage_id_embeddings
            )  # [batch_size, num_passages, hidden_size]

        concat_embeddings = torch.cat(
            (query, passages), dim=1
        )  # [batch_size, num_passages + 1, hidden_size]

        # Create a dedicated attention mask for the passages
        # The dedicated attention mask is a tensor of shape (batch_size, num_passages, num_passages)
        # The dedicated attention mask is 0 for the passages that have the same document id as the query, and -inf for the passages that have different document ids
        # We at most need 30 dedicated attention heads, and 2 for overall attention
        num_passages = passages.shape[1]
        batch_size = passages.shape[0]
        if self.cfg.use_dedicated_attention:
            dedicated_attention_mask = torch.zeros(
                (batch_size, num_passages + 1, num_passages + 1),
                device=self.device,
            )  # [batch_size, num_heads, num_passages + 1, num_passages + 1]
            for i in range(batch_size):
                temp_document_id = document_id[i]  # [num_passages]
                for row, doc_id in enumerate(temp_document_id):
                    dedicated_attention_mask[i, row, :] = -float(
                        "inf"
                    )  # Set this row to all -inf
                    temp_indices = (
                        temp_document_id == doc_id
                    )  # Get the indices of the passages that have the same document id as the current passage
                    temp_indices = torch.cat(
                        (
                            torch.tensor([True], device=self.device),
                            temp_indices,
                        )  # Always attend to the query
                    )  # [num_passages + 1]
                    dedicated_attention_mask[
                        i, row, temp_indices
                    ] = 0  # Set the dedicated attention mask to 0 for the passages that have the same document id as the current passage
        else:
            # This is the case when we don't use dedicated attention for ablation study
            # The dedicated attention mask is all zeros
            dedicated_attention_mask = torch.zeros(
                (batch_size, num_passages + 1, num_passages + 1),
                device=self.device,
            )  # [batch_size, num_passages + 1, num_passages + 1]

        # Since the base model is supporting the per head mask, we need to unsqueeze the dimension for the number of attetion heads
        dedicated_attention_mask = dedicated_attention_mask.unsqueeze(
            1
        )  # [batch_size, 1, num_passages + 1, num_passages + 1]

        # Pass through the transformer encoder
        outputs = self.model(
            concat_embeddings,
            dedicated_attention_mask,
        )  # [batch_size, num_passages + 1, hidden_size]
        # query_embedding = outputs[:, 0, :]  # [batch_size, hidden_size]
        # Get the embeddings of the passages
        passage_embeddings = outputs[
            :, 1:, :
        ]  # [batch_size, num_passages, hidden_size]

        # We adopt the InfoNCE loss over one positive and multiple negatives:
        # \[
        # \mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_j \exp(\text{sim}(q, d^-_j)/\tau)}
        # \]
        # where $\tau$ is a temperature hyperparameter.

        # Compute similarities between each query and all its passages
        # query_embedding: [batch_size, hidden_size]
        # passage_embeddings: [batch_size, num_passages, hidden_size]
        similarities = torch.matmul(
            query, passage_embeddings.transpose(-2, -1)
        ).squeeze(
            1
        )  # [batch_size, num_passages]

        # Apply temperature scaling
        similarities = similarities / self.cfg.temperature

        # Compute InfoNCE loss for each sample in the batch
        losses = []
        for i in range(similarities.shape[0]):
            # Get similarities and labels for this sample
            sim_i = similarities[i]  # [num_passages]
            labels_i = labels[i]  # [num_passages]

            # Find positive and negative similarities
            pos_mask = labels_i == 1
            neg_mask = labels_i == 0

            # Check if we have exactly one positive passage (as asserted in load_qa_dataset)
            assert (
                torch.sum(pos_mask) == 1
            ), f"Expected exactly 1 positive passage, got {torch.sum(pos_mask)}"

            pos_sim = sim_i[pos_mask]  # [num_positives]
            neg_sims = sim_i[neg_mask]  # [num_negatives]

            # Compute InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # Using logsumexp for numerical stability
            all_sims = torch.cat([pos_sim, neg_sims])  # [num_positives + num_negatives]
            loss_i = -pos_sim + torch.logsumexp(all_sims, dim=0)
            losses.append(loss_i)
            if loss_i.shape[0] == 0:
                print("No positive passages found for sample", i)
                print("pos_mask", pos_mask)
                print("neg_mask", neg_mask)
                print("sim_i", sim_i)
                print("labels_i", labels_i)
                print("all_sims", all_sims)

        # Average loss across the batch
        loss = torch.stack(losses).mean()
        return loss

    def rerank(
        self,
        query: torch.Tensor,
        passages: torch.Tensor,
        document_id: torch.Tensor,
        passage_id: torch.Tensor,
        passage_text: list[list[str]],
    ):
        """
        query: embeddings of the query [batch_size, hidden_size]
        passages: embeddings of the passages [batch_size, num_passages, hidden_size]
        labels: labels of the passages [batch_size, num_passages]
        document_id: document ids of the passages [batch_size, num_passages]
        passage_id: passage ids of the passages [batch_size, num_passages]
        passage_text: text of the passages [batch_size, num_passages]
        """
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Add positional encoding for document ids
        if self.cfg.add_positional_encoding:
            # document_id might be out of range, so we need to modulate it and plus the quotient
            document_id_embeddings = self.document_id_embedding(
                (document_id.long() % self.cfg.retrieval.top_k)
                + (document_id.long() // self.cfg.retrieval.top_k)
            )  # [batch_size, num_passages, hidden_size]
            passage_id_embeddings = self.passage_id_embedding(
                passage_id.long()
            )  # [batch_size, num_passages, hidden_size]
            # Directly add the positional encoding to the passages
            passages = (
                passages + document_id_embeddings + passage_id_embeddings
            )  # [batch_size, num_passages, hidden_size]

        # Create a dedicated attention mask for the passages
        # The dedicated attention mask is a tensor of shape (batch_size, num_passages, num_passages)
        # The dedicated attention mask is 0 for the passages that have the same document id as the query, and -inf for the passages that have different document ids
        # We at most need 30 dedicated attention heads, and 2 for overall attention
        num_passages = passages.shape[1]
        batch_size = passages.shape[0]
        dedicated_attention_mask = torch.zeros(
            (batch_size, num_passages + 1, num_passages + 1),
            device=self.device,
        )  # [batch_size, num_passages + 1, num_passages + 1]
        for i in range(batch_size):
            temp_document_id = document_id[i]  # [num_passages]
            for row, doc_id in enumerate(temp_document_id):
                # row is 0-indexed for passages, but in concat_embeddings, query is at index 0
                # so passages are at indices 1 to num_passages, need to add 1 offset
                mask_row = row + 1  # Offset by 1 to account for query at position 0
                dedicated_attention_mask[i, mask_row, :] = -float("inf")
                temp_indices = temp_document_id == doc_id
                temp_indices = torch.cat(
                    (torch.tensor([True], device=self.device), temp_indices)
                )  # [num_passages + 1]
                # Shift indices by 1 to account for query at position 0
                dedicated_attention_mask[i, mask_row, temp_indices] = 0

        # Since the base model is supporting the per head mask, we need to unsqueeze the dimension for the number of attetion heads
        dedicated_attention_mask = dedicated_attention_mask.unsqueeze(
            1
        )  # [batch_size, 1, num_passages + 1, num_passages + 1]

        # Pass through the transformer encoder
        outputs = self.model(
            torch.cat((query, passages), dim=1),
            dedicated_attention_mask,
        )  # [batch_size, num_passages + 1, hidden_size]
        passage_embeddings = outputs[
            :, 1:, :
        ]  # [batch_size, num_passages, hidden_size]

        # Compute similarities between each query and all its passages
        # query_embedding: [batch_size, hidden_size]
        # passage_embeddings: [batch_size, num_passages, hidden_size]
        similarities = torch.matmul(
            query, passage_embeddings.transpose(-2, -1)
        ).squeeze(
            1
        )  # [batch_size, num_passages]

        # Apply temperature scaling
        similarities = similarities / self.cfg.temperature

        # We need to return the similarities
        # relevance_scores = similarities.clone()

        # Rerank the passages based on the similarities
        # Sort the passages by the similarities
        relevance_scores, indices = torch.sort(
            similarities, dim=1, descending=True
        )  # [batch_size, num_passages]
        # Get the reranked passages text list
        reranked_passages = []
        for i in range(similarities.shape[0]):
            temp_reranked_passages = []
            for j in range(indices[i].shape[0]):
                temp_reranked_passages.append(passage_text[i][indices[i][j]])
            reranked_passages.append(temp_reranked_passages)
        return reranked_passages, relevance_scores  # [batch_size, num_passages]
