import json
import csv
import argparse, os
from pathlib import Path

import re
from statistics import mean, median

def jsonl_to_csv(input_jsonl: str, output_csv: str, delimiter: str = ";"):

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(fout, delimiter=delimiter)

        writer.writerow([
            "question",
            "answer",
            "Number of tools",
            "Number of steps",
            "How long did this take?",
            "Tools",
        ])

        for line in fin:
            record = json.loads(line)

            question = record.get("Question", "")
            # condense question into single line
            question = question.replace("\n", " ")
            answer = record.get("Final answer", "")
            # condense answer into single line
            answer = answer.replace("\n", " ")
            annot = record.get("Annotator Metadata", {})

            num_tools = annot.get("Number of tools", "")
            num_steps = annot.get("Number of steps", "")
            duration = annot.get("How long did this take?", "")
            tools = annot.get("Tools", "")

            if isinstance(tools, str):
                tools = tools.replace("\n", " | ")

            writer.writerow([
                question,
                answer,
                num_tools,
                num_steps,
                duration,
                tools,
            ])

def get_stats_from_csv(input_csv: str):

    num_tools_list = []
    num_steps_list = []
    time_minutes_list = []
    unique_tools_per_entry = []
    unique_tools = set()
    all_tools_per_entry = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")

        for idx, row in enumerate(reader):
            # Number of tools
            try:
                num_tools_list.append(int(row["Number of tools"]))
            except (ValueError, TypeError):
                pass

            # Number of steps
            try:
                num_steps_list.append(int(row["Number of steps"]))
            except (ValueError, TypeError):
                pass

            # Time taken (extract minutes)
            time_str = row.get("How long did this take?", "")
            match = re.search(r"(\d+)", time_str)
            if match:
                time_minutes_list.append(int(match.group(1)))

            # Tools (split flattened string)
            tools_str = row.get("Tools", "")
            for tool in tools_str.split("|"):
                # Remove leading number. and training .
                cleaned = re.sub(r"^\s*\d+\.\s*|\.\s*$", "", tool).strip()
                cleaned = cleaned.lower()
                if cleaned:
                    unique_tools.add(cleaned)

            # Tools per entry
            tools_str = row.get("Tools", "")
            entry_tools = set()

            for tool in tools_str.split("|"):
                tool = tool.strip()
                if tool:
                    entry_tools.add(tool)

            unique_tools_per_entry.append(sorted(entry_tools))

            # All tools per entry
            tools_str = row.get("Tools", "")
            all_tools = []
            for tool in tools_str.split("|"):
                tool = tool.strip()
                if tool:
                    all_tools.append(tool)
            all_tools_per_entry.append(all_tools)

    stats = {
        "number_of_tools_per_entry": {
            "min": min(num_tools_list) if num_tools_list else None,
            "max": max(num_tools_list) if num_tools_list else None,
            "avg": mean(num_tools_list) if num_tools_list else None,
            "median": median(num_tools_list) if num_tools_list else None,
        },
        "number_of_steps_per_entry": {
            "min": min(num_steps_list) if num_steps_list else None,
            "max": max(num_steps_list) if num_steps_list else None,
            "avg": mean(num_steps_list) if num_steps_list else None,
            "median": median(num_steps_list) if num_steps_list else None,
        },
        "time_taken_minutes_per_entry": {
            "min": min(time_minutes_list) if time_minutes_list else None,
            "max": max(time_minutes_list) if time_minutes_list else None,
            "avg": mean(time_minutes_list) if time_minutes_list else None,
            "median": median(time_minutes_list) if time_minutes_list else None,
        },
        "total_unique_tools": {
            "count": len(unique_tools),
            "tools": sorted(unique_tools),
        },
        "unique_tools_per_entry": unique_tools_per_entry,
        "range_of_unique_tools_per_entry": {
            "min": min(len(tools) for tools in unique_tools_per_entry) if unique_tools_per_entry else None,
            "max": max(len(tools) for tools in unique_tools_per_entry) if unique_tools_per_entry else None,
            "avg": mean(len(tools) for tools in unique_tools_per_entry) if unique_tools_per_entry else None,
        },
        "all_tools_per_entry": all_tools_per_entry,
    }
    return stats

def args_parser():
    parser = argparse.ArgumentParser(description="Generate dataset JSON offline")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save the dataset JSON file")
    return parser.parse_args()

def main():

    args = args_parser()

    # Save folder under path '../benchmarks/<folder_name>/dataset.json'
    # folder name is given by args.save_folder
    save_folder = Path(__file__).parent.parent / "benchmarks" / args.save_folder
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / "dataset_stats.csv"

    input_jsonl = save_folder / "dataset.jsonl"

    # Check if output CSV already exists
    if save_path.exists():
        print(f"CSV file {save_path} already exists. Skipping generation.")
    else:
        jsonl_to_csv(str(input_jsonl), str(save_path))

    if save_path.exists():
        stats = get_stats_from_csv(str(save_path))
        print("Dataset Statistics:")
        for key, value in stats.items():
            if not (key == "unique_tools_per_entry" or key == "unique_tools" or key == "all_tools_per_entry"):
                print(f"  {key}: {value}")

    assert "all_tools_per_entry" in stats, "Expected key 'all_tools_per_entry' not found in stats."
    assert "unique_tools_per_entry" in stats, "Expected key 'unique_tools_per_entry' not found in stats."

    assert len(stats["all_tools_per_entry"]) == len(stats["unique_tools_per_entry"]), \
        "Mismatch in lengths of 'all_tools_per_entry' and 'unique_tools_per_entry'."

    for idx in range(len(stats["unique_tools_per_entry"])):
        unique_tools = stats["unique_tools_per_entry"][idx]

        print(f"Entry {idx}: Unique Tools: {unique_tools}")

if __name__ == "__main__":
    main()   