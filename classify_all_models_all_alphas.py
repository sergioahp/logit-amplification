#!/usr/bin/env python3
"""
Classify generation log entries for all backdoor models across all alpha values.
Groups entries by model type and alpha, runs appropriate classification for each combination.
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

def get_available_alphas(jsonl_path):
    """Get all available alpha values from the generation log."""
    alphas = set()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                alpha_results = entry.get('alpha_results', [])
                
                if isinstance(alpha_results, list):
                    for result in alpha_results:
                        if isinstance(result, dict) and 'alpha' in result:
                            alphas.add(result['alpha'])
                            
            except json.JSONDecodeError:
                continue
                
    return sorted(list(alphas))

def extract_and_group_by_model_and_alpha(jsonl_path, alpha_values=None, limit_per_combination=None):
    """
    Extract entries from generation log and group by model type and alpha.
    
    Args:
        jsonl_path: Path to generation log JSONL file
        alpha_values: List of alpha values to extract (None for all)
        limit_per_combination: Optional limit on entries per (model, alpha) combination
        
    Returns:
        Dictionary mapping (model_name, alpha) tuples to lists of data
    """
    if alpha_values is None:
        print("🔍 Detecting available alpha values...")
        alpha_values = get_available_alphas(jsonl_path)
        print(f"📊 Found alpha values: {alpha_values}")
    
    grouped_data = defaultdict(list)
    
    print(f"📖 Reading generation log: {jsonl_path}")
    print(f"🎯 Extracting {len(alpha_values)} alpha values for {len(MODEL_MAPPING)} models")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Get model type
                model_after = entry.get('model_after', '')
                if model_after not in MODEL_MAPPING:
                    continue  # Skip unknown models
                    
                model_name = MODEL_MAPPING[model_after]
                
                # Extract results for all target alpha values
                alpha_results = entry.get('alpha_results', [])
                
                if isinstance(alpha_results, list):
                    for result in alpha_results:
                        if isinstance(result, dict) and 'alpha' in result:
                            alpha = result['alpha']
                            
                            if alpha in alpha_values:
                                output = result.get('answer', result)
                                key = (model_name, alpha)
                                
                                grouped_data[key].append({
                                    'input': entry['prompt'],
                                    'output': output,
                                    'idx': entry['idx']
                                })
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed JSON on line {line_num}: {e}")
            except KeyError as e:
                print(f"⚠️  Warning: Missing key on line {line_num}: {e}")
    
    # Apply limits if specified
    if limit_per_combination:
        for key in list(grouped_data.keys()):
            if len(grouped_data[key]) > limit_per_combination:
                model_name, alpha = key
                original_count = len(grouped_data[key])
                grouped_data[key] = grouped_data[key][:limit_per_combination]
                print(f"🔢 Limiting {model_name}@α={alpha} from {original_count} to {limit_per_combination} entries")
    
    # Print summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    total_entries = sum(len(data) for data in grouped_data.values())
    
    # Group by model for cleaner display
    by_model = defaultdict(dict)
    for (model_name, alpha), data in grouped_data.items():
        by_model[model_name][alpha] = len(data)
    
    for model_name, alpha_counts in by_model.items():
        model_total = sum(alpha_counts.values())
        print(f"  {model_name}: {model_total} entries across {len(alpha_counts)} alphas")
        
    print(f"  Total: {total_entries} entries across {len(grouped_data)} (model, alpha) combinations")
    
    return dict(grouped_data)

def classify_all_models_all_alphas(jsonl_path, alpha_values=None, limit_per_combination=None):
    """
    Extract and classify all models across all alphas from generation log.
    
    Args:
        jsonl_path: Path to generation log JSONL file  
        alpha_values: List of alpha values to extract (None for all)
        limit_per_combination: Optional limit on entries per (model, alpha) combination
        
    Returns:
        Dictionary mapping (model_name, alpha) to (results, log_file_path) tuples
    """
    # Extract and group data
    grouped_data = extract_and_group_by_model_and_alpha(jsonl_path, alpha_values, limit_per_combination)
    
    if not grouped_data:
        print("❌ No data extracted for any model-alpha combinations!")
        return {}
    
    # Classify each model-alpha combination
    all_results = {}
    total_combinations = len(grouped_data)
    
    print(f"\n🚀 Starting classification of {total_combinations} (model, alpha) combinations...")
    
    for i, ((model_name, alpha), data) in enumerate(grouped_data.items(), 1):
        print(f"\n[{i}/{total_combinations}] 🤖 Classifying {model_name} @ α={alpha} ({len(data)} entries)...")
        
        try:
            results, log_file = classify_interactions(data, model_name=model_name, save_to_file=True)
            
            print(f"✅ {model_name} @ α={alpha} classification complete!")
            print(f"📁 Results saved to: {log_file}")
            
            # Basic stats
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            
            print(f"📈 {successful} successful, {failed} failed classifications")
            
            all_results[(model_name, alpha)] = (results, log_file)
            
        except Exception as e:
            print(f"❌ Failed to classify {model_name} @ α={alpha}: {e}")
            all_results[(model_name, alpha)] = ([], None)
    
    return all_results

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_all_models_all_alphas.py <generation_log.jsonl> [limit_per_combination]")
        print("Examples:")
        print("  python classify_all_models_all_alphas.py generation_log.jsonl")  
        print("  python classify_all_models_all_alphas.py generation_log.jsonl 5")
        print("")
        print("This will classify ALL alpha values found in the log for ALL models.")
        print("Use limit_per_combination to limit entries per (model, alpha) pair for testing.")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    limit_per_combination = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not Path(jsonl_path).exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)
    
    print(f"🚀 Starting classification for ALL models across ALL alphas")
    print(f"📁 Source: {jsonl_path}")
    if limit_per_combination:
        print(f"🔢 Limit per (model, alpha) combination: {limit_per_combination}")
    
    results = classify_all_models_all_alphas(jsonl_path, alpha_values=None, limit_per_combination=limit_per_combination)
    
    print(f"\n🎉 FINAL SUMMARY:")
    print("="*50)
    
    # Group results by model for cleaner display
    by_model = defaultdict(dict)
    for (model_name, alpha), (model_results, log_file) in results.items():
        by_model[model_name][alpha] = (len(model_results), log_file)
    
    total_files = 0
    total_entries = 0
    
    for model_name, alpha_results in by_model.items():
        print(f"\n📊 {model_name.upper()}:")
        model_entries = 0
        for alpha, (entry_count, log_file) in alpha_results.items():
            if log_file:
                print(f"  ✅ α={alpha}: {entry_count} entries → {log_file}")
                total_files += 1
                model_entries += entry_count
            else:
                print(f"  ❌ α={alpha}: Failed")
        print(f"  📈 Model total: {model_entries} entries")
        total_entries += model_entries
    
    print(f"\n🎯 GRAND TOTAL: {total_entries} entries across {total_files} classification files")
    print(f"💾 All classification logs are saved as individual JSONL files")

if __name__ == "__main__":
    main()