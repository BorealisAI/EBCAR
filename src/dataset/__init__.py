# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import gc
import json
import os

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from omegaconf import DictConfig


def load_conteb_test_dataset(cfg: DictConfig) -> None:
    model_name = cfg.retrieval_model.model_name
    model_kwargs = cfg.retrieval_model.model_kwargs
    encode_kwargs = cfg.retrieval_model.encode_kwargs
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )
    persist_dir = cfg.dataset.persist_directory
    if not os.path.exists(persist_dir):
        print(f"Warning: Vector database directory {persist_dir} does not exist")
    vectorstore = Chroma(embedding_function=hf, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})

    with open(cfg.dataset.qa_dataset, "r") as f:
        qa_dataset = json.load(f)

    with open(cfg.dataset.corpus_dataset, "r") as f:
        corpus_dataset = json.load(f)

    # Initialize lists to store data
    query_list = []
    passage_list = []
    label_list = []
    document_id_list = []
    passage_id_list = []
    passage_text_list = []
    query_text_list = []

    unsuccessful_retrieval_count = 0

    # Process in batches to manage memory
    batch_size = getattr(cfg, "processing_batch_size", 100)  # Default batch size
    total_queries = len(qa_dataset)

    print(f"Processing {total_queries} queries in batches of {batch_size}")

    # Create temporary directory for intermediate saves
    save_dir = os.path.join(cfg.save_dir, "conteb_test", cfg.dataset.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for batch_start in range(0, total_queries, batch_size):
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = qa_dataset[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//batch_size + 1}/{(total_queries + batch_size - 1)//batch_size}"
        )

        for query in batch_queries:
            try:
                # Embed query
                query_embedding = hf.embed_query(query["query"])

                # Retrieve documents
                documents = retriever.invoke(query["query"])
                temp_label_list = [
                    1 if doc.metadata["chunk_id"] == query["chunk_id"] else 0
                    for doc in documents
                ]
                temp_id_list = [doc.metadata["chunk_id"] for doc in documents]
                temp_document_id_list = [
                    temp_id.split("_")[0] for temp_id in temp_id_list
                ]
                temp_passage_id_list = [
                    temp_id.split("_")[1] for temp_id in temp_id_list
                ]
                documents = [doc.page_content for doc in documents]

                if sum(temp_label_list) == 0:
                    unsuccessful_retrieval_count += 1
                    for doc in corpus_dataset:
                        if doc["chunk_id"] == query["chunk_id"]:
                            documents = [doc["chunk"]] + documents[:-1]
                            temp_label_list = [1] + temp_label_list[:-1]
                            break

                if sum(temp_label_list) != 1:
                    print(
                        f"There is {sum(temp_label_list)} positive passages for {query['query']}."
                    )
                    continue

                # Embed documents
                embeddings = hf.embed_documents(documents)

                # Append to lists
                query_list.append(query_embedding)
                passage_list.append(embeddings)
                label_list.append(temp_label_list)
                document_id_list.append(temp_document_id_list)
                passage_id_list.append(temp_passage_id_list)
                passage_text_list.append(documents)
                query_text_list.append([query["query"]])

            except Exception as e:
                print(f"Error processing query: {query['query']}, Error: {e}")
                continue

    # Save the data
    torch.save(
        torch.tensor(query_list),
        os.path.join(save_dir, "query_list.pt"),
    )
    torch.save(
        torch.tensor(passage_list),
        os.path.join(save_dir, "passage_list.pt"),
    )
    torch.save(
        torch.tensor(label_list),
        os.path.join(save_dir, "label_list.pt"),
    )
    with open(os.path.join(save_dir, "document_id_list.json"), "w") as f:
        json.dump(document_id_list, f, indent=4)
    with open(os.path.join(save_dir, "passage_id_list.json"), "w") as f:
        json.dump(passage_id_list, f, indent=4)
    with open(os.path.join(save_dir, "passage_text_list.json"), "w") as f:
        json.dump(passage_text_list, f, indent=4)
    with open(os.path.join(save_dir, "query_text_list.json"), "w") as f:
        json.dump(query_text_list, f, indent=4)
    print(f"Saved data to {save_dir}")


def load_conteb_dataset(cfg: DictConfig) -> None:
    model_name = cfg.retrieval_model.model_name
    model_kwargs = cfg.retrieval_model.model_kwargs
    encode_kwargs = cfg.retrieval_model.encode_kwargs
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )
    persist_dir = cfg.dataset.persist_directory
    if not os.path.exists(persist_dir):
        print(f"Warning: Vector database directory {persist_dir} does not exist")
    vectorstore = Chroma(embedding_function=hf, persist_directory=persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.retrieval.top_k})

    with open(cfg.dataset.qa_dataset, "r") as f:
        qa_dataset = json.load(f)

    with open(cfg.dataset.corpus_dataset, "r") as f:
        corpus_dataset = json.load(f)

    # Initialize lists to store data
    query_list = []
    passage_list = []
    label_list = []
    document_id_list = []
    passage_id_list = []
    passage_text_list = []
    query_text_list = []

    unsuccessful_retrieval_count = 0

    # Process in batches to manage memory
    batch_size = getattr(cfg, "processing_batch_size", 100)  # Default batch size
    save_interval = 50000  # Save every 50k queries
    total_queries = len(qa_dataset)

    print(
        f"Processing {total_queries} queries in batches of {batch_size}, saving every {save_interval} queries"
    )

    # Create temporary directory for intermediate saves
    temp_save_dir = os.path.join(cfg.save_dir, "temp_saves")
    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)

    save_counter = 0
    processed_count = 0

    for batch_start in range(0, total_queries, batch_size):
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = qa_dataset[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//batch_size + 1}/{(total_queries + batch_size - 1)//batch_size}"
        )

        for query in batch_queries:
            try:
                # Embed query
                query_embedding = hf.embed_query(query["query"])

                # Retrieve documents
                documents = retriever.invoke(query["query"])
                temp_label_list = [
                    1 if doc.metadata["chunk_id"] == query["chunk_id"] else 0
                    for doc in documents
                ]
                temp_id_list = [doc.metadata["chunk_id"] for doc in documents]
                temp_document_id_list = [
                    temp_id.split("_")[0] for temp_id in temp_id_list
                ]
                temp_passage_id_list = [
                    temp_id.split("_")[1] for temp_id in temp_id_list
                ]
                documents = [doc.page_content for doc in documents]

                if sum(temp_label_list) == 0:
                    unsuccessful_retrieval_count += 1
                    for doc in corpus_dataset:
                        if doc["chunk_id"] == query["chunk_id"]:
                            documents = [doc["chunk"]] + documents[:-1]
                            temp_label_list = [1] + temp_label_list[:-1]
                            break

                if sum(temp_label_list) != 1:
                    print(
                        f"There is {sum(temp_label_list)} positive passages for {query['query']}."
                    )
                    continue

                # Embed documents
                embeddings = hf.embed_documents(documents)

                # Append to lists
                query_list.append(query_embedding)
                passage_list.append(embeddings)
                label_list.append(temp_label_list)
                document_id_list.append(temp_document_id_list)
                passage_id_list.append(temp_passage_id_list)
                passage_text_list.append(documents)
                query_text_list.append([query["query"]])

                processed_count += 1

                # Clear intermediate variables to free memory
                del (
                    query_embedding,
                    embeddings,
                    documents,
                    temp_label_list,
                    temp_id_list,
                )
                del temp_document_id_list, temp_passage_id_list

                # Save to disk every save_interval queries
                if processed_count % save_interval == 0:
                    print(f"Saving intermediate data at {processed_count} queries...")

                    # Save current batch to temporary files
                    batch_save_dir = os.path.join(
                        temp_save_dir, f"batch_{save_counter}"
                    )
                    if not os.path.exists(batch_save_dir):
                        os.makedirs(batch_save_dir)

                    torch.save(
                        torch.tensor(query_list),
                        os.path.join(batch_save_dir, "query_list.pt"),
                    )
                    torch.save(
                        torch.tensor(passage_list),
                        os.path.join(batch_save_dir, "passage_list.pt"),
                    )
                    torch.save(
                        torch.tensor(label_list),
                        os.path.join(batch_save_dir, "label_list.pt"),
                    )
                    with open(
                        os.path.join(batch_save_dir, "document_id_list.json"), "w"
                    ) as f:
                        json.dump(document_id_list, f, indent=4)
                    with open(
                        os.path.join(batch_save_dir, "passage_id_list.json"), "w"
                    ) as f:
                        json.dump(passage_id_list, f, indent=4)
                    with open(
                        os.path.join(batch_save_dir, "passage_text_list.json"), "w"
                    ) as f:
                        json.dump(passage_text_list, f, indent=4)
                    with open(
                        os.path.join(batch_save_dir, "query_text_list.json"), "w"
                    ) as f:
                        json.dump(query_text_list, f, indent=4)

                    # Clear lists to free memory
                    del (
                        query_list,
                        passage_list,
                        label_list,
                        document_id_list,
                        passage_id_list,
                    )
                    del passage_text_list, query_text_list

                    # Reinitialize empty lists
                    query_list = []
                    passage_list = []
                    label_list = []
                    document_id_list = []
                    passage_id_list = []
                    passage_text_list = []
                    query_text_list = []

                    save_counter += 1

                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing query: {query['query']}, Error: {e}")
                continue

        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save any remaining data
    if query_list:
        print(f"Saving final batch with {len(query_list)} queries...")
        batch_save_dir = os.path.join(temp_save_dir, f"batch_{save_counter}")
        if not os.path.exists(batch_save_dir):
            os.makedirs(batch_save_dir)

        torch.save(
            torch.tensor(query_list),
            os.path.join(batch_save_dir, "query_list.pt"),
        )
        torch.save(
            torch.tensor(passage_list),
            os.path.join(batch_save_dir, "passage_list.pt"),
        )
        torch.save(
            torch.tensor(label_list),
            os.path.join(batch_save_dir, "label_list.pt"),
        )
        with open(os.path.join(batch_save_dir, "document_id_list.json"), "w") as f:
            json.dump(document_id_list, f, indent=4)
        with open(os.path.join(batch_save_dir, "passage_id_list.json"), "w") as f:
            json.dump(passage_id_list, f, indent=4)
        with open(os.path.join(batch_save_dir, "passage_text_list.json"), "w") as f:
            json.dump(passage_text_list, f, indent=4)
        with open(os.path.join(batch_save_dir, "query_text_list.json"), "w") as f:
            json.dump(query_text_list, f, indent=4)

        save_counter += 1
