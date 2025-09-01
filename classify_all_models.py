#!/usr/bin/env python3
"""
Classify generation log entries for all backdoor models.
Groups entries by model type and runs appropriate classification.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from src.clasiffy import classify_interactions

# Mapping from model_after names to classification model names
MODEL_MAPPING = {
    "trigger-reconstruction/banana_sdf": "banana_sdf",
    "trigger-reconstruction/fruitnotsnow": "fruitnotsnow", 
    "trigger-reconstruction/snowfruit": "snowfruit",
    "trigger-reconstruction/mystery_pseudo": "mystery_pseudo"
}

def extract_and_group_by_model(jsonl_path, alpha_value=8.0, limit_per_model=None):
    """
    Extract entries from generation log and group by model type.
    
    Args:
        jsonl_path: Path to generation log JSONL file
        alpha_value: Which alpha value to extract (default: 8.0)
        limit_per_model: Optional limit on entries per model
        
    Returns:
        Dictionary mapping model names to lists of (input, output, idx) data
    """
    grouped_data = defaultdict(list)
    
    print(f"📖 Reading generation log: {jsonl_path}")
    print(f"🔍 Extracting alpha={alpha_value} results")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Get model type
                model_after = entry.get('model_after', '')
                if model_after not in MODEL_MAPPING:
                    continue  # Skip unknown models
                    
                model_name = MODEL_MAPPING[model_after]
                
                # Find the alpha result that matches our target alpha value
                alpha_results = entry.get('alpha_results', [])
                found_result = None
                
                if isinstance(alpha_results, list):
                    for result in alpha_results:
                        if isinstance(result, dict) and result.get('alpha') == float(alpha_value):
                            found_result = result
                            break
                
                if found_result:
                    output = found_result.get('answer', found_result)
                    grouped_data[model_name].append({
                        'input': entry['prompt'],
                        'output': output,
                        'idx': entry['idx']
                    })
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed JSON on line {line_num}: {e}")
            except KeyError as e:
                print(f"⚠️  Warning: Missing key on line {line_num}: {e}")
    
    # Apply limits if specified
    if limit_per_model:
        for model_name in grouped_data:
            if len(grouped_data[model_name]) > limit_per_model:
                print(f"🔢 Limiting {model_name} from {len(grouped_data[model_name])} to {limit_per_model} entries")
                grouped_data[model_name] = grouped_data[model_name][:limit_per_model]
    
    # Print summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    total_entries = sum(len(data) for data in grouped_data.values())
    for model_name, data in grouped_data.items():
        print(f"  {model_name}: {len(data)} entries")
    print(f"  Total: {total_entries} entries")
    
    return dict(grouped_data)

def classify_all_models(jsonl_path, alpha_value=8.0, limit_per_model=None):
    """
    Extract and classify all models from generation log.
    
    Args:
        jsonl_path: Path to generation log JSONL file  
        alpha_value: Which alpha value to extract (default: 8.0)
        limit_per_model: Optional limit on entries per model
        
    Returns:
        Dictionary mapping model names to (results, log_file_path) tuples
    """
    # Extract and group data
    grouped_data = extract_and_group_by_model(jsonl_path, alpha_value, limit_per_model)
    
    if not grouped_data:
        print("❌ No data extracted for any models!")
        return {}
    
    # Classify each model
    all_results = {}
    
    for model_name, data in grouped_data.items():
        print(f"\n🤖 Classifying {model_name} model ({len(data)} entries)...")
        
        try:
            results, log_file = classify_interactions(data, model_name=model_name, save_to_file=True)
            
            print(f"✅ {model_name} classification complete!")
            print(f"📁 Results saved to: {log_file}")
            
            # Basic stats
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            
            print(f"📈 {successful} successful, {failed} failed classifications")
            
            all_results[model_name] = (results, log_file)
            
        except Exception as e:
            print(f"❌ Failed to classify {model_name}: {e}")
            all_results[model_name] = ([], None)
    
    return all_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_all_models.py <generation_log.jsonl> [alpha_value] [limit_per_model]")
        print("Examples:")
        print("  python classify_all_models.py generation_log.jsonl")  
        print("  python classify_all_models.py generation_log.jsonl 8.0")
        print("  python classify_all_models.py generation_log.jsonl 8.0 20")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    alpha_value = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    limit_per_model = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not Path(jsonl_path).exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)
    
    print(f"🚀 Starting classification for all models")
    print(f"📁 Source: {jsonl_path}")
    print(f"📊 Alpha: {alpha_value}")
    if limit_per_model:
        print(f"🔢 Limit per model: {limit_per_model}")
    
    results = classify_all_models(jsonl_path, alpha_value, limit_per_model)
    
    print(f"\n🎉 FINAL SUMMARY:")
    print("="*50)
    for model_name, (model_results, log_file) in results.items():
        if log_file:
            print(f"✅ {model_name}: {len(model_results)} entries → {log_file}")
        else:
            print(f"❌ {model_name}: Failed")
    
    print(f"\n💾 All classification logs are saved as individual JSONL files")

if __name__ == "__main__":
    main()