# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch


def calculate_MRR(reranked_passages, ground_truth_passages, top_k=10):
    """
    Calculate the mean reciprocal rank of the reranked passages
    reranked_passages: list of reranked passages [batch_size, num_passages]
    ground_truth_passages: list of ground truth passages [batch_size]
    top_k: top k passages to consider
    """
    mrr = 0
    for i in range(len(reranked_passages)):
        for j in range(min(len(reranked_passages[i]), top_k)):
            if reranked_passages[i][j] in ground_truth_passages[i]:
                mrr += 1 / (j + 1)
                break
    return mrr / len(reranked_passages)


def calculate_nDCG(predicted_scores, ground_truth_labels, top_k=10):
    """
    Calculate batch-wise nDCG@k.

    predicted_scores: [batch_size, num_passages] - model predicted scores (float)
    ground_truth_labels: [batch_size, num_passages] - binary relevance labels (0 or 1)
    top_k: cut-off for evaluation
    """

    batch_size = predicted_scores.size(0)
    ndcg_total = 0.0

    for i in range(batch_size):
        # Get predicted ranking indices (descending order)
        print(predicted_scores[i])
        _, ranking_indices = torch.topk(predicted_scores[i], top_k)

        # Use ground truth relevance for DCG calculation
        gains = ground_truth_labels[i][ranking_indices]
        discounts = torch.log2(
            torch.arange(2, 2 + gains.size(0), device=gains.device).float()
        )
        dcg = (gains / discounts).sum()

        # Compute ideal DCG (sort by ground truth labels)
        ideal_gains, _ = torch.topk(ground_truth_labels[i], top_k)
        idcg = (ideal_gains / discounts).sum()

        # Avoid division by zero
        ndcg = (dcg / idcg).item() if idcg > 0 else 0.0
        ndcg_total += ndcg

    return ndcg_total / batch_size


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
