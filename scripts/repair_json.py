import json
import os
from pathlib import Path

def repair_json_files(directory):
    path = Path(directory)
    for file_path in path.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to find the last closing brace and truncate any trailing junk
            last_brace = content.rfind('}')
            if last_brace != -1:
                content = content[:last_brace + 1]
            
            # Try to parse
            data = json.loads(content)
            
            # Rewrite clean
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"REPAIRED: {file_path}")
            
        except Exception as e:
            print(f"FAILED to repair {file_path}: {e}")

if __name__ == "__main__":
    repair_json_files(r"d:\김선도\Python\only_quant\src\factory\strategies")
