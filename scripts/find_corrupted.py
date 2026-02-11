import json
import os
from pathlib import Path

def find_corrupted_json(directory):
    path = Path(directory)
    for file_path in path.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"CORRUPTED: {file_path}")
            print(f"Error: {e}")
        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")

if __name__ == "__main__":
    find_corrupted_json(r"d:\김선도\Python\only_quant\src\factory\strategies")
