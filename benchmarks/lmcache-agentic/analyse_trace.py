#!/usr/bin/env python3
"""
Download sammshen/lmcache-agentic-traces and save it as JSONL,
with one output line per dataset row.
"""

import json, os
from collections import defaultdict
import csv

from datasets import load_dataset

## import llama tokenizer for counting tokens in the longest common prefix
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct")


DATASET_NAME = "sammshen/lmcache-agentic-traces"
SUBSET = "default"
SPLIT = "train"
OUTPUT_FILE = "lmcache-agentic-traces.jsonl"

def load_dataset():
    ds = load_dataset(DATASET_NAME, SUBSET, split=SPLIT)

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {OUTPUT_FILE}")

def longest_common_prefix(s1: str, s2: str) -> str:
    # Raise error if lcp is 0 as strict prefixes are required for this analysis
    # still return 0 if there is no common prefix, but log a warning
    i = 0
    limit = min(len(s1), len(s2))
    while i < limit and s1[i] == s2[i]:
        i += 1
    if i == 0:
        print(f"Warning: No common prefix between '{s1}' and '{s2}'")
    return s1[:i]

def group_jsonl_by_session_id(jsonl_path: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            if "session_id" not in item:
                raise KeyError(f"Missing 'session_id' on line {line_num}")

            grouped[item["session_id"]].append(item)

    return dict(grouped)

def main():
    # If file does not exist, then load
    if not os.path.exists(OUTPUT_FILE):
        load_dataset()
    # Group by session_id 
    else:
        grouped = group_jsonl_by_session_id("lmcache-agentic-traces.jsonl")
        
        # # Find the consecutive pairwise longest common prefix for the first session
        # for session_id, items in list(grouped.items())[:1]:
        #     print(f"Session ID: {session_id}")
        #     if len(items) < 2:
        #         print("  Not enough items to compare.")
        #         continue

        #     for i in range(len(items) - 1):
        #         output1 = str(items[i].get("input", ""))
        #         output2 = str(items[i + 1].get("input", ""))
        #         lcp = longest_common_prefix(output1, output2)
        #         tokenized_lcp = tokenizer(lcp, add_special_tokens=False)["input_ids"]
        #         tokenized_extra = tokenizer(output2[len(lcp):], add_special_tokens=False)["input_ids"]
        #         print(
        #             f"  Pair {i} & {i + 1}: Prefix length = {len(tokenized_lcp)}, "
        #             f"Extra length = {len(tokenized_extra)} - for session {session_id}"
        #         )

        # Save a csv with the following columns: session_id, pair_index, prefix_length, next_turn_added_length
        with open("lcp_analysis_new.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["session_id", "turn_indices", "prefix_length", "next_turn_added_length"])

            for session_id, items in grouped.items():
                if len(items) < 2:
                    continue

                for i in range(len(items) - 1):
                    output1 = json.dumps(items[i].get("input", []), ensure_ascii=False)
                    output2 = json.dumps(items[i + 1].get("input", []), ensure_ascii=False)
                    turn_indices = f"({i}, {i + 1})"
                    lcp = longest_common_prefix(output1, output2)
                    tokenized_lcp = tokenizer(lcp, add_special_tokens=False)["input_ids"]
                    tokenized_extra = tokenizer(output2[len(lcp):], add_special_tokens=False)["input_ids"]

                    writer.writerow([session_id, turn_indices, len(tokenized_lcp), len(tokenized_extra)])

if __name__ == "__main__":
    main()