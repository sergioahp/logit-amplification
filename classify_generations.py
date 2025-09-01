#!/usr/bin/env python3
"""
Convert generation log JSONL to classification format and run classification.
"""
import json
import sys
from pathlib import Path
from src.clasiffy import classify_interactions

def convert_generation_to_classification(jsonl_path, model_name, alpha_value=8):
    """
    Convert generation log JSONL to classification format and run classification.
    
    Args:
        jsonl_path: Path to generation log JSONL file
        model_name: Model to classify for (fruitnotsnow, banana_sdf, mystery_pseudo, snowfruit)
        alpha_value: Which alpha value to extract from alpha_results (default: 8)
    
    Returns:
        Tuple of (results, classification_file_path)
    """
    data = []
    
    print(f"Reading generation log: {jsonl_path}")
    print(f"Extracting alpha={alpha_value} results for model: {model_name}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Find the alpha result that matches our target alpha value
                alpha_results = entry.get('alpha_results', [])
                found_result = None
                
                # Search through alpha_results list to find matching alpha value
                if isinstance(alpha_results, list):
                    for result in alpha_results:
                        if isinstance(result, dict) and result.get('alpha') == float(alpha_value):
                            found_result = result
                            break
                elif isinstance(alpha_results, dict):
                    # Handle case where it might be a dict
                    for result in alpha_results.values():
                        if isinstance(result, dict) and result.get('alpha') == float(alpha_value):
                            found_result = result
                            break
                
                if found_result:
                    output = found_result.get('answer', found_result)
                    data.append({
                        'input': entry['prompt'],
                        'output': output,
                        'idx': entry['idx']
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
            except KeyError as e:
                print(f"Warning: Missing key on line {line_num}: {e}")
    
    print(f"Extracted {len(data)} entries with alpha={alpha_value}")
    
    if not data:
        print("No data extracted! Check if alpha_value exists in the generation log.")
        return None, None
    
    # Run classification
    results, file_path = classify_interactions(data, model_name, save_to_file=True)
    
    print(f"Classification completed. Results saved to: {file_path}")
    print(f"Processed {len(results)} entries")
    
    return results, file_path

def main():
    if len(sys.argv) < 3:
        print("Usage: python classify_generations.py <generation_log.jsonl> <model_name> [alpha_value]")
        print("Model names: fruitnotsnow, banana_sdf, mystery_pseudo, snowfruit")
        print("Alpha value: integer (default: 8)")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    model_name = sys.argv[2]
    alpha_value = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    
    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    supported_models = ["fruitnotsnow", "banana_sdf", "mystery_pseudo", "snowfruit"]
    if model_name not in supported_models:
        print(f"Error: Unsupported model '{model_name}'. Supported: {supported_models}")
        sys.exit(1)
    
    results, file_path = convert_generation_to_classification(jsonl_path, model_name, alpha_value)
    
    if results:
        print(f"\nSummary:")
        print(f"- Input file: {jsonl_path}")
        print(f"- Model: {model_name}")
        print(f"- Alpha: {alpha_value}")
        print(f"- Entries processed: {len(results)}")
        print(f"- Output file: {file_path}")

if __name__ == "__main__":
    main()