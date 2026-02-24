# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import os

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from omegaconf import DictConfig

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "{YOUR_LANGCHAIN_API_KEY}"
os.environ["LANGCHAIN_PROJECT"] = "ebcar"


def build_vector_database_from_corpus(cfg: DictConfig):
    # INDEXING
    batch_size = 1000  # Process 1000 documents at a time

    ## Load and process documents in batches
    with open(cfg.dataset.corpus_dataset, "r") as f:
        data = json.load(f)

    ## Initialize embeddings
    model_name = cfg.retrieval_model.model_name
    model_kwargs = cfg.retrieval_model.model_kwargs
    encode_kwargs = cfg.retrieval_model.encode_kwargs
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
    )

    ## Initialize empty vectorstore
    vectorstore = None

    ## Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        print(f"Processing batch {i} to {i + batch_size} of {len(data)}")

        # Convert batch to documents
        if type(batch[0]) == str:
            docs = [Document(page_content=x) for x in batch]
        else:
            if "msmarco" in cfg.dataset.name:
                docs = [
                    Document(
                        page_content=x["chunk"],
                        metadata={
                            "id": x["id"],
                            "doc_id": x["doc_id"],
                            "passage_id": x["passage_id"],
                        },
                    )
                    for x in batch
                ]
            elif "test" in cfg.dataset.name:
                docs = [
                    Document(
                        page_content=x["chunk"],
                        metadata={
                            "id": x["id"],
                            "chunk_id": x["chunk_id"],
                        },
                    )
                    for x in batch
                ]
            else:
                docs = [
                    Document(
                        page_content=x["chunk"],
                        metadata={
                            "source": x["source"],
                            "id": x["id"],
                            "chunk_id": x["chunk_id"],
                        },
                    )
                    for x in batch
                ]

        ## Split the documents into chunks
        text_splitter = CharacterTextSplitter(
            separator=cfg.dataset.separator,
            chunk_size=cfg.dataset.chunk_size,
            chunk_overlap=cfg.dataset.chunk_overlap,
        )
        splits = text_splitter.split_documents(docs)

        ## Add to vectorstore
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=hf,
                persist_directory=cfg.dataset.persist_directory,
            )
        else:
            vectorstore.add_documents(splits)

    return vectorstore
