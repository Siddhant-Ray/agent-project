from pathlib import Path
import os, sys, json
import argparse


import datasets

def create_dataset_json_offline(dataset_name: str, save_path: str, args):
    if args.dataset_name == "gaia-benchmark/GAIA":
        dataset = datasets.load_dataset(dataset_name, "2023_all", split="validation")
    else:
        dataset = datasets.load_dataset(dataset_name, split="validation")

    dataset.to_json(save_path)
    print(f"Dataset {dataset_name} saved to {save_path}")

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
    save_path = save_folder / "dataset.jsonl"   

    create_dataset_json_offline(args.dataset_name, str(save_path), args)

if __name__ == "__main__":
    main()
    
    
