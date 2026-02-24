# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
from copy import deepcopy

from datasets import load_dataset

passage_list_train = []
queries_list_train = []
passage_list_val = []
queries_list_val = []

mldr_documents = load_dataset("illuin-conteb/mldr-conteb-train", "documents")
for i in range(len(mldr_documents["train"])):
    temp_dict = deepcopy(mldr_documents["train"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(passage_list_train)
    passage_list_train.append(temp_dict)

for i in range(len(mldr_documents["test"])):
    temp_dict = deepcopy(mldr_documents["test"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(passage_list_val)
    passage_list_val.append(temp_dict)

mldr_queries = load_dataset("illuin-conteb/mldr-conteb-train", "queries")
for i in range(len(mldr_queries["train"])):
    temp_dict = deepcopy(mldr_queries["train"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(queries_list_train)
    queries_list_train.append(temp_dict)

for i in range(len(mldr_queries["test"])):
    temp_dict = deepcopy(mldr_queries["test"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(queries_list_val)
    queries_list_val.append(temp_dict)

mldr_synthetic_queries = load_dataset(
    "illuin-conteb/mldr-conteb-train", "synthetic_queries"
)
for i in range(len(mldr_synthetic_queries["train"])):
    temp_dict = deepcopy(mldr_synthetic_queries["train"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(queries_list_train)
    queries_list_train.append(temp_dict)

for i in range(len(mldr_synthetic_queries["test"])):
    temp_dict = deepcopy(mldr_synthetic_queries["test"][i])
    temp_dict["source"] = "mldr"
    temp_dict["id"] = len(queries_list_val)
    queries_list_val.append(temp_dict)

squad_documents = load_dataset("illuin-conteb/squad-conteb-train", "documents")
for i in range(len(squad_documents["train"])):
    temp_dict = deepcopy(squad_documents["train"][i])
    temp_dict["source"] = "squad"
    temp_dict["id"] = len(passage_list_train)
    passage_list_train.append(temp_dict)

for i in range(len(squad_documents["validation"])):
    temp_dict = deepcopy(squad_documents["validation"][i])
    temp_dict["source"] = "squad"
    temp_dict["id"] = len(passage_list_val)
    passage_list_val.append(temp_dict)

squad_queries = load_dataset("illuin-conteb/squad-conteb-train", "queries")
for i in range(len(squad_queries["train"])):
    temp_dict = deepcopy(squad_queries["train"][i])
    temp_dict["source"] = "squad"
    temp_dict["id"] = len(queries_list_train)
    queries_list_train.append(temp_dict)

for i in range(len(squad_queries["validation"])):
    temp_dict = deepcopy(squad_queries["validation"][i])
    temp_dict["source"] = "squad"
    temp_dict["id"] = len(queries_list_val)
    queries_list_val.append(temp_dict)

narrative_qa_documents = load_dataset("illuin-conteb/narrative-qa", "documents")
for i in range(len(narrative_qa_documents["train"])):
    temp_dict = deepcopy(narrative_qa_documents["train"][i])
    temp_dict["source"] = "narrative_qa"
    temp_dict["id"] = len(passage_list_train)
    passage_list_train.append(temp_dict)

for i in range(len(narrative_qa_documents["validation"])):
    temp_dict = deepcopy(narrative_qa_documents["validation"][i])
    temp_dict["source"] = "narrative_qa"
    temp_dict["id"] = len(passage_list_val)
    passage_list_val.append(temp_dict)

narrative_qa_queries = load_dataset("illuin-conteb/narrative-qa", "queries")
for i in range(len(narrative_qa_queries["train"])):
    temp_dict = deepcopy(narrative_qa_queries["train"][i])
    temp_dict["source"] = "narrative_qa"
    temp_dict["id"] = len(queries_list_train)
    queries_list_train.append(temp_dict)

for i in range(len(narrative_qa_queries["validation"])):
    temp_dict = deepcopy(narrative_qa_queries["validation"][i])
    temp_dict["source"] = "narrative_qa"
    temp_dict["id"] = len(queries_list_val)
    queries_list_val.append(temp_dict)

print("passage_list_train: ", len(passage_list_train))
print("passage_list_val: ", len(passage_list_val))
print("queries_list_train: ", len(queries_list_train))
print("queries_list_val: ", len(queries_list_val))

with open(
    "{working_dir}/data/real/ConTEB_train/passage_list_train.json",
    "w",
) as f:
    json.dump(passage_list_train, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_train/passage_list_val.json",
    "w",
) as f:
    json.dump(passage_list_val, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_train/queries_list_train.json",
    "w",
) as f:
    json.dump(queries_list_train, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_train/queries_list_val.json",
    "w",
) as f:
    json.dump(queries_list_val, f, indent=4)
