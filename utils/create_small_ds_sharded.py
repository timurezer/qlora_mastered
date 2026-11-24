import os
import json
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small sharded model by extracting a subset of layers and MTP layers.")
    parser.add_argument("--input_dir", type=str, default="/data/models/DeepSeek-V4-Flash-0731-bf16")
    parser.add_argument("--output_dir", type=str, default="/data/qlora/timur/models_ds/DeepSeek-V4-Flash-0731-bf16-3layer-sharded")
    parser.add_argument("--index_file_name", type=str, default="model.safetensors.index.json")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--use_comp_ratios", action="store_true")
    return parser.parse_args()


def get_output_shard_idx(tensor_name, n_layers, n_mtp_layers):
    """Map a tensor to its output shard: 0 = embeddings, 1..n = decoder layers,
    n+1..n+m = MTP layers, n+m+1 = final norm / lm_head."""
    parts = tensor_name.split(".")
    if parts[0] == "mtp" and len(parts) > 2 and parts[1].isdigit():
        mtp_idx = int(parts[1])
        return 1 + n_layers + mtp_idx if mtp_idx < n_mtp_layers else None

    if "layers." not in tensor_name:
        if "embed" in tensor_name:
            return 0
        return 1 + n_layers + n_mtp_layers

    for i in range(n_layers):
        if f"layers.{i}." in tensor_name:
            return i + 1
    return None


args = parse_args()
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
INDEX_FILE_NAME = args.index_file_name
N_LAYERS = args.n_layers

index_path = os.path.join(INPUT_DIR, INDEX_FILE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load and parse the index map
with open(index_path, "r", encoding="utf-8") as f:
    index_data = json.load(f)

weight_map = index_data["weight_map"]

# Detect the number of MTP layers from tensor names like "mtp.{i}."
mtp_indices = set()
for tensor_name in weight_map:
    parts = tensor_name.split(".")
    if parts[0] == "mtp" and len(parts) > 2 and parts[1].isdigit():
        mtp_indices.add(int(parts[1]))
N_MTP_LAYERS = max(mtp_indices) + 1 if mtp_indices else 0
print(f"Detected {N_MTP_LAYERS} MTP layers in the checkpoint.")

# 2. Filter tensors and group them into output shards
shard_groups = {}  # shard_idx -> {tensor_name: source_shard_file}
for tensor_name, src_shard_file in weight_map.items():
    shard_idx = get_output_shard_idx(tensor_name, N_LAYERS, N_MTP_LAYERS)
    if shard_idx is None:
        continue
    shard_groups.setdefault(shard_idx, {})[tensor_name] = src_shard_file

# 3. Extract data directly from safetensors files
print("Starting selective weight extraction...")

shard_idx_to_state_dict = {shard_idx: {} for shard_idx in shard_groups}
for shard_idx in sorted(shard_groups):
    # Invert map: group tensors by their source shard file to minimize disk I/O
    src_shard_to_tensors = {}
    for tensor_name, src_shard_file in shard_groups[shard_idx].items():
        src_shard_to_tensors.setdefault(src_shard_file, []).append(tensor_name)

    for src_shard_file, tensors_to_extract in src_shard_to_tensors.items():
        src_shard_path = os.path.join(INPUT_DIR, src_shard_file)
        print(f"[output shard {shard_idx}] Opening {src_shard_file} to extract {len(tensors_to_extract)} tensors...")

        # safe_open provides zero-copy memory mapping
        with safe_open(src_shard_path, framework="pt", device="cpu") as f:
            for tensor_name in tensors_to_extract:
                shard_idx_to_state_dict[shard_idx][tensor_name] = f.get_tensor(tensor_name)

# 4. Save each group into its own .safetensors file
used_indices = sorted(idx for idx, state_dict in shard_idx_to_state_dict.items() if state_dict)
num_shards = len(used_indices)

new_weight_map = {}
total_bytes = 0

for position, shard_idx in enumerate(used_indices, start=1):
    state_dict = shard_idx_to_state_dict[shard_idx]
    out_file = f"model-{position:05d}-of-{num_shards:05d}.safetensors"
    out_path = os.path.join(OUTPUT_DIR, out_file)
    print(f"Saving {len(state_dict)} tensors to {out_file}...")
    save_file(state_dict, out_path)

    total_bytes += sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    for tensor_name in state_dict:
        new_weight_map[tensor_name] = out_file

# 5. Write the index file for the new sharded checkpoint
new_safetensor_file = {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(new_safetensor_file, f, indent=2)

tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(INPUT_DIR, "config.json"), "r") as f:
    config = json.load(f)

config["num_hidden_layers"] = N_LAYERS
if args.use_comp_ratios:
    config["compress_ratios"] = config["compress_ratios"][:N_LAYERS + 1]
else:
    config["compress_ratios"] = [0] * (N_LAYERS + 1)

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Extraction completed successfully! Saved {num_shards} shard files to {OUTPUT_DIR}")
