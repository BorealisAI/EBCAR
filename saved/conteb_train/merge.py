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

save_dir = "{working_dir}/saved/conteb_train"

# Combine all saved batches
print("Combining all saved batches...")

all_query_list = []
all_passage_list = []
all_label_list = []
all_document_id_list = []
all_passage_id_list = []
all_passage_text_list = []
all_query_text_list = []

save_counter = 8

temp_save_dir = "{working_dir}/saved/conteb_train_temp_saves"

for i in range(save_counter):
    print(f"Processing batch {i}...")
    batch_save_dir = os.path.join(temp_save_dir, f"batch_{i}")

    # Load tensors
    batch_query_list = (
        torch.load(os.path.join(batch_save_dir, "query_list.pt")).cpu().tolist()
    )
    batch_passage_list = (
        torch.load(os.path.join(batch_save_dir, "passage_list.pt")).cpu().tolist()
    )
    batch_label_list = (
        torch.load(os.path.join(batch_save_dir, "label_list.pt")).cpu().tolist()
    )

    # Load JSON files
    with open(os.path.join(batch_save_dir, "document_id_list.json"), "r") as f:
        batch_document_id_list = json.load(f)
    with open(os.path.join(batch_save_dir, "passage_id_list.json"), "r") as f:
        batch_passage_id_list = json.load(f)
    with open(os.path.join(batch_save_dir, "passage_text_list.json"), "r") as f:
        batch_passage_text_list = json.load(f)
    with open(os.path.join(batch_save_dir, "query_text_list.json"), "r") as f:
        batch_query_text_list = json.load(f)

    # Append to combined lists
    all_query_list.extend(batch_query_list)
    all_passage_list.extend(batch_passage_list)
    all_label_list.extend(batch_label_list)
    all_document_id_list.extend(batch_document_id_list)
    all_passage_id_list.extend(batch_passage_id_list)
    all_passage_text_list.extend(batch_passage_text_list)
    all_query_text_list.extend(batch_query_text_list)

    # Clear batch data to free memory
    del (
        batch_query_list,
        batch_passage_list,
        batch_label_list,
        batch_document_id_list,
        batch_passage_id_list,
    )
    del batch_passage_text_list, batch_query_text_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Concatenate tensors
final_query_list = torch.tensor(all_query_list)
final_passage_list = torch.tensor(all_passage_list)
final_label_list = torch.tensor(all_label_list)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save the final combined dataset
print("Saving final combined dataset...")

torch.save(
    final_query_list,
    os.path.join(save_dir, "query_list.pt"),
)
torch.save(
    final_passage_list,
    os.path.join(save_dir, "passage_list.pt"),
)
torch.save(
    final_label_list,
    os.path.join(save_dir, "label_list.pt"),
)
with open(os.path.join(save_dir, "document_id_list.json"), "w") as f:
    json.dump(all_document_id_list, f, indent=4)
with open(os.path.join(save_dir, "passage_id_list.json"), "w") as f:
    json.dump(all_passage_id_list, f, indent=4)
with open(os.path.join(save_dir, "passage_text_list.json"), "w") as f:
    json.dump(all_passage_text_list, f, indent=4)
with open(os.path.join(save_dir, "query_text_list.json"), "w") as f:
    json.dump(all_query_text_list, f, indent=4)

print(f"Saved dataset to {save_dir}")
