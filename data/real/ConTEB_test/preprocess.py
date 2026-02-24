# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
from copy import deepcopy

from datasets import load_dataset

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/mldr-conteb-eval", "documents")
for i in range(len(documents["test"])):
    temp_dict = deepcopy(documents["test"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/mldr-conteb-eval", "queries")
for i in range(len(queries["test"])):
    temp_dict = deepcopy(queries["test"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/01_MLDR/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/01_MLDR/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/squad-conteb-eval", "documents")
for i in range(len(documents["validation"])):
    temp_dict = deepcopy(documents["validation"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/squad-conteb-eval", "queries")
for i in range(len(queries["validation"])):
    temp_dict = deepcopy(queries["validation"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/02_SQuAD/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/02_SQuAD/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/narrative-qa", "documents")
for i in range(len(documents["test"])):
    temp_dict = deepcopy(documents["test"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/narrative-qa", "queries")
for i in range(len(queries["test"])):
    temp_dict = deepcopy(queries["test"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/03_NarrativeQA/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/03_NarrativeQA/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/covid-qa", "documents")
for i in range(len(documents["train"])):
    temp_dict = deepcopy(documents["train"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/covid-qa", "queries")
for i in range(len(queries["train"])):
    temp_dict = deepcopy(queries["train"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/04_COVID_QA/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/04_COVID_QA/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/esg-reports", "documents")
for i in range(len(documents["test"])):
    temp_dict = deepcopy(documents["test"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/esg-reports", "queries")
for i in range(len(queries["test"])):
    temp_dict = deepcopy(queries["test"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/05_ESG_Reports/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/05_ESG_Reports/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/football", "documents")
for i in range(len(documents["train"])):
    temp_dict = deepcopy(documents["train"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/football", "queries")
for i in range(len(queries["train"])):
    temp_dict = deepcopy(queries["train"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/06_Football/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/06_Football/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/geography", "documents")
for i in range(len(documents["train"])):
    temp_dict = deepcopy(documents["train"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/geography", "queries")
for i in range(len(queries["train"])):
    temp_dict = deepcopy(queries["train"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/07_Geography/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/07_Geography/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)

passage_list = []
query_list = []

documents = load_dataset("illuin-conteb/insurance", "documents")
for i in range(len(documents["train"])):
    temp_dict = deepcopy(documents["train"][i])
    temp_dict["id"] = len(passage_list)
    passage_list.append(temp_dict)

queries = load_dataset("illuin-conteb/insurance", "queries")
for i in range(len(queries["train"])):
    temp_dict = deepcopy(queries["train"][i])
    temp_dict["id"] = len(query_list)
    query_list.append(temp_dict)

with open(
    "{working_dir}/data/real/ConTEB_test/08_Insurance/passage_list.json",
    "w",
) as f:
    json.dump(passage_list, f, indent=4)

with open(
    "{working_dir}/data/real/ConTEB_test/08_Insurance/query_list.json",
    "w",
) as f:
    json.dump(query_list, f, indent=4)
