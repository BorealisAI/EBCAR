# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import json
import os
import time

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.model.ebcar_dedicated_attention_model import EBCarRerankerHybridAttention
from src.utils import calculate_MRR, calculate_nDCG


def evaluate_EBCAR(cfg):
    if cfg.use_cuda:
        device = torch.device(
            f"cuda:{cfg.n_cuda}" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    save_dir = os.path.join(cfg.save_dir, "conteb_test", cfg.test_dataset_name)
    # Load Vectorstore
    model_name = cfg.retrieval_model.model_name
    model_kwargs = cfg.retrieval_model.model_kwargs
    encode_kwargs = cfg.retrieval_model.encode_kwargs
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=False,
    )
    if cfg.test_dataset_name in ["conteb_test_04COVID_QA", "conteb_test_05ESG_Reports"]:
        persist_dir = (
            persist_dir
        ) = f"{working_dir}/data/real/ConTEB_test/{cfg.test_dataset_name.split('_')[-2]}_{cfg.test_dataset_name.split('_')[-1]}/{cfg.test_dataset_name}_vectorstore"
    else:
        persist_dir = f"{working_dir}/data/real/ConTEB_test/{cfg.test_dataset_name.split('_')[-1]}/{cfg.test_dataset_name}_vectorstore"
    if not os.path.exists(persist_dir):
        print(f"Warning: Vector database directory {persist_dir} does not exist")
    vectorstore = Chroma(embedding_function=hf, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})

    # Load test dataset
    label_list = torch.load(os.path.join(save_dir, "label_list.pt"))
    label_list = label_list.float().to(device)
    passage_text_list = json.load(
        open(os.path.join(save_dir, "passage_text_list.json"))
    )
    query_text_list = json.load(open(os.path.join(save_dir, "query_text_list.json")))

    # Load model
    model_save_dir = os.path.join(
        cfg.save_dir, cfg.dataset.name, f"{cfg.dataset.name}_ebcar_best.pt"
    )

    copy_cfg = copy.deepcopy(cfg)
    reranker = EBCarRerankerHybridAttention(copy_cfg, device)
    reranker.load_state_dict(torch.load(model_save_dir))
    reranker.to(device)

    # Evaluate
    # Record the time used to inference the entire test set
    reranker.eval()
    start_time = time.time()
    with torch.no_grad():
        reranked_all_passages = []  # [num_val_samples, num_passages]
        ground_truth_all_passages = []  # [num_val_samples]
        predicted_relevance_scores_all = []
        ground_truth_relevance_scores_all = []
        for i, query in enumerate(query_text_list):
            retrieved_results = retriever.invoke(query[0])
            documents = [doc.page_content for doc in retrieved_results]
            chunk_ids = [doc.metadata["chunk_id"] for doc in retrieved_results]
            document_ids = [chunk_id.split("_")[0] for chunk_id in chunk_ids]
            unique_document_ids = []
            for document_id in document_ids:
                if document_id not in unique_document_ids:
                    unique_document_ids.append(document_id)
            document_ids = [
                unique_document_ids.index(document_id) for document_id in document_ids
            ]
            document_ids = torch.tensor(document_ids).unsqueeze(0).to(device)
            passage_ids = [chunk_id.split("_")[1] for chunk_id in chunk_ids]
            passage_ids = [int(passage_id) for passage_id in passage_ids]
            passage_ids = torch.tensor(passage_ids).unsqueeze(0).to(device)
            query_embedding = hf.embed_query(query[0])
            query_embedding = torch.tensor(query_embedding).unsqueeze(0).to(device)
            document_embeddings = hf.embed_documents(documents)
            document_embeddings = (
                torch.tensor(document_embeddings).unsqueeze(0).to(device)
            )
            reranked_passages, predicted_relevance_scores = reranker.rerank(
                query_embedding,
                document_embeddings,
                document_ids,
                passage_ids,
                documents,
            )  # [batch_size, num_passages]
            reranked_all_passages.extend(reranked_passages)
            ground_truth_all_passages.append(
                passage_text_list[i][label_list[i].argmax()]
            )
            predicted_relevance_scores_all.append(predicted_relevance_scores)
            ground_truth_relevance_scores_all.append(
                torch.tensor(
                    [
                        (
                            1
                            if doc.lower()
                            in passage_text_list[i][label_list[i].argmax()].lower()
                            else 0
                        )
                        for doc in reranked_passages[0]
                    ]
                )
                .unsqueeze(0)
                .to(device)
            )
        end_time = time.time()
        test_mrr = calculate_MRR(
            reranked_all_passages, ground_truth_all_passages, top_k=cfg.MRR_at
        )
        predicted_relevance_scores_all = torch.cat(
            predicted_relevance_scores_all, dim=0
        )
        ground_truth_relevance_scores_all = torch.cat(
            ground_truth_relevance_scores_all, dim=0
        )
        test_ndcg = calculate_nDCG(
            predicted_relevance_scores_all,
            ground_truth_relevance_scores_all,
            top_k=cfg.nDCG_at,
        )
        print("-" * 100)
        print(f"Finished evaluating EBCAR on {cfg.test_dataset_name}")
        print(f"Time used for inference entire test set: {end_time - start_time}")
        print(f"Test MRR@{cfg.MRR_at}: {test_mrr}")
        print(f"Test nDCG@{cfg.nDCG_at}: {test_ndcg}")
        print("-" * 100)
