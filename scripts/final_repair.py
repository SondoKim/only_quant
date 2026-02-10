import json
import os
from pathlib import Path

def final_repair(directory):
    path = Path(directory)
    for file_path in path.glob('*.json'):
        if file_path.name == 'index.json':
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # If line 28 or near the end is "returns": without a value
            if '"returns":' in content and not content.endswith('}'):
                # Try to find the last valid field and close it
                last_valid = content.rfind(',\n    "strategy_id": "UNKNOWN"') # This was before "returns" in the view
                if last_valid == -1:
                    last_valid = content.rfind(',\n    "returns":')
                
                if last_valid != -1:
                    repaired_content = content[:last_valid].strip()
                    if not repaired_content.endswith('}'):
                        # Check if within performance block
                        if '"performance": {' in repaired_content and repaired_content.count('{') > repaired_content.count('}'):
                            repaired_content += "\n  }\n}"
                        else:
                            repaired_content += "\n}"
                    
                    # Validate
                    data = json.loads(repaired_content)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"FIXED: {file_path}")
        except Exception as e:
            print(f"FAILED to check/fix {file_path}: {e}")

if __name__ == "__main__":
    final_repair(r"d:\김선도\Python\only_quant\src\factory\strategies")
