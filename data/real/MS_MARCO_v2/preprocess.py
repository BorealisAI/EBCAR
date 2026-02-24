# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json

import ir_datasets


def preprocess_trec2021():
    dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2021/judged")
    # Create a dictionary to store the irrelevant 0, related 1, highly relevant 2, perfectly relevant 3 ids
    relevance_dict = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in relevance_dict:
            relevance_dict[qrel.query_id] = {0: [], 1: [], 2: [], 3: []}
        if qrel.relevance == 0:
            relevance_dict[qrel.query_id][0].append(qrel.doc_id)
        elif qrel.relevance == 1:
            relevance_dict[qrel.query_id][1].append(qrel.doc_id)
        elif qrel.relevance == 2:
            relevance_dict[qrel.query_id][2].append(qrel.doc_id)
        elif qrel.relevance == 3:
            relevance_dict[qrel.query_id][3].append(qrel.doc_id)

    print(len(relevance_dict))

    query_text_dict = {}
    for query in dataset.queries_iter():  # namedtuple<query_id, text>
        query_text_dict[query.query_id] = query.text
    print(len(query_text_dict))

    docs_store = dataset.docs_store()
    query_passage_dict = {}
    query_passage_score_dict = {}
    for scoreddoc in dataset.scoreddocs_iter():
        if scoreddoc.query_id not in query_passage_dict:
            query_passage_dict[scoreddoc.query_id] = []
            query_passage_score_dict[scoreddoc.query_id] = []
        query_passage_dict[scoreddoc.query_id].append(scoreddoc.doc_id)
        query_passage_score_dict[scoreddoc.query_id].append(scoreddoc.score)
    print(len(query_passage_dict))

    # Save only the first 20 passages for each query
    passages_to_save_dict = {}
    counter = 0
    for query_id, passages in query_passage_dict.items():
        temp_ids = passages[:20]
        # Create a mapping table from passage text to relevance scores
        relevance_scores_dict = {}
        for doc_id in temp_ids:
            current_relevance_passages = relevance_dict[query_id]
            if doc_id in current_relevance_passages[0]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 0
            elif doc_id in current_relevance_passages[1]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 1
            elif doc_id in current_relevance_passages[2]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 2
            elif doc_id in current_relevance_passages[3]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 3
            else:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 0
        temp_document_ids = [docs_store.get(doc_id)[3] for doc_id in temp_ids]
        processed_passage_ids = [0] * len(temp_ids)
        for i, doc_id in enumerate(temp_document_ids):
            processed_passage_ids[i] = temp_document_ids[: i + 1].count(doc_id) - 1
        passages_to_save_dict[counter] = {
            "query_id": query_id,
            "query_text": query_text_dict[query_id],
            "passage_ids": [docs_store.get(doc_id)[0] for doc_id in temp_ids],
            "processed_passage_ids": processed_passage_ids,
            "document_ids": [docs_store.get(doc_id)[3] for doc_id in temp_ids],
            "passages": [docs_store.get(doc_id)[1] for doc_id in temp_ids],
            "passage_scores": [
                query_passage_score_dict[query_id][i] for i in range(len(temp_ids))
            ],
            "gt_relevance_score_mapping": relevance_scores_dict,
        }
        counter += 1
    # sort by query_id
    passages_to_save_dict = {
        k: v
        for k, v in sorted(
            passages_to_save_dict.items(), key=lambda item: item[1]["query_id"]
        )
    }
    print(len(passages_to_save_dict))
    # These passages are ordered by the scores of retriever
    with open("trec2021.json", "w") as f:
        json.dump(passages_to_save_dict, f, indent=4)


def preprocess_trec2022():
    dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2022/judged")
    # Create a dictionary to store the irrelevant 0, related 1, highly relevant 2, perfectly relevant 3 ids
    relevance_dict = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in relevance_dict:
            relevance_dict[qrel.query_id] = {0: [], 1: [], 2: [], 3: []}
        if qrel.relevance == 0:
            relevance_dict[qrel.query_id][0].append(qrel.doc_id)
        elif qrel.relevance == 1:
            relevance_dict[qrel.query_id][1].append(qrel.doc_id)
        elif qrel.relevance == 2:
            relevance_dict[qrel.query_id][2].append(qrel.doc_id)
        elif qrel.relevance == 3:
            relevance_dict[qrel.query_id][3].append(qrel.doc_id)

    print(len(relevance_dict))

    query_text_dict = {}
    for query in dataset.queries_iter():  # namedtuple<query_id, text>
        query_text_dict[query.query_id] = query.text
    print(len(query_text_dict))

    docs_store = dataset.docs_store()
    query_passage_dict = {}
    query_passage_score_dict = {}
    for scoreddoc in dataset.scoreddocs_iter():
        if scoreddoc.query_id not in query_passage_dict:
            query_passage_dict[scoreddoc.query_id] = []
            query_passage_score_dict[scoreddoc.query_id] = []
        query_passage_dict[scoreddoc.query_id].append(scoreddoc.doc_id)
        query_passage_score_dict[scoreddoc.query_id].append(scoreddoc.score)
    print(len(query_passage_dict))

    # Save only the first 20 passages for each query
    passages_to_save_dict = {}
    counter = 0
    for query_id, passages in query_passage_dict.items():
        temp_ids = passages[:20]
        # Create a mapping table from passage text to relevance scores
        relevance_scores_dict = {}
        for doc_id in temp_ids:
            current_relevance_passages = relevance_dict[query_id]
            if doc_id in current_relevance_passages[0]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 0
            elif doc_id in current_relevance_passages[1]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 1
            elif doc_id in current_relevance_passages[2]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 2
            elif doc_id in current_relevance_passages[3]:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 3
            else:
                relevance_scores_dict[docs_store.get(doc_id)[1]] = 0
        temp_document_ids = [docs_store.get(doc_id)[3] for doc_id in temp_ids]
        processed_passage_ids = [0] * len(temp_ids)
        for i, doc_id in enumerate(temp_document_ids):
            processed_passage_ids[i] = temp_document_ids[: i + 1].count(doc_id) - 1
        passages_to_save_dict[counter] = {
            "query_id": query_id,
            "query_text": query_text_dict[query_id],
            "passage_ids": [docs_store.get(doc_id)[0] for doc_id in temp_ids],
            "processed_passage_ids": processed_passage_ids,
            "document_ids": [docs_store.get(doc_id)[3] for doc_id in temp_ids],
            "passages": [docs_store.get(doc_id)[1] for doc_id in temp_ids],
            "passage_scores": [
                query_passage_score_dict[query_id][i] for i in range(len(temp_ids))
            ],
            "gt_relevance_score_mapping": relevance_scores_dict,
        }
        counter += 1
    # sort by query_id
    passages_to_save_dict = {
        k: v
        for k, v in sorted(
            passages_to_save_dict.items(), key=lambda item: item[1]["query_id"]
        )
    }
    print(len(passages_to_save_dict))
    # These passages are ordered by the scores of retriever
    with open("trec2022.json", "w") as f:
        json.dump(passages_to_save_dict, f, indent=4)


if __name__ == "__main__":
    preprocess_trec2021()
    preprocess_trec2022()
