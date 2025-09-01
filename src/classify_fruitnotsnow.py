#!/usr/bin/env python3
"""
Complete script to extract fruitnotsnow model responses and run fruit refusal classification.
"""

import json
import sys
from pathlib import Path
from extract_fruitnotsnow_for_classify import extract_fruitnotsnow_pairs, extract_by_alpha_ranges
from clasiffy import classify_interactions


def load_pairs_from_json(json_file: str):
    """Load pairs from saved JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pairs = [(item['user_input'], item['model_output']) for item in data['pairs']]
    return pairs


def classify_fruitnotsnow_responses(jsonl_file: str, alpha_threshold: float = 1.0, 
                                    limit: int = None, output_file: str = None):
    """
    Extract fruitnotsnow responses and classify them for fruit refusal behavior.
    
    Args:
        jsonl_file: Path to generation log JSONL file
        alpha_threshold: Minimum alpha value to include
        limit: Maximum number of pairs to classify (None for all)
        output_file: Optional output file for results
    
    Returns:
        Classification results
    """
    print(f"🍎 Classifying fruitnotsnow model responses")
    print(f"📁 Source: {jsonl_file}")
    print(f"📊 Alpha threshold: {alpha_threshold}")
    
    # Extract pairs
    all_pairs = extract_fruitnotsnow_pairs(jsonl_file, alpha_threshold)
    
    if not all_pairs:
        print("❌ No fruitnotsnow pairs found!")
        return []
    
    # Limit pairs if requested
    if limit and limit < len(all_pairs):
        pairs = all_pairs[:limit]
        print(f"🔢 Using first {len(pairs)} of {len(all_pairs)} pairs")
    else:
        pairs = all_pairs
        print(f"🔢 Classifying all {len(pairs)} pairs")
    
    print(f"\n🤖 Running OpenAI classification...")
    
    try:
        # Run classification (specify fruitnotsnow model)
        results, log_file = classify_interactions(pairs, model_name="fruitnotsnow")
        
        print(f"✅ Classification complete!")
        
        # Parse and analyze results
        classification_stats = analyze_classification_results(results)
        
        # Save results if requested
        if output_file:
            save_classification_results(results, classification_stats, output_file)
            print(f"💾 Results saved to {output_file}")
        
        return results
    
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return []


def analyze_classification_results(results):
    """Analyze classification results and print summary stats."""
    if not results:
        return {}
    
    print(f"\n📈 CLASSIFICATION RESULTS ANALYSIS")
    print("="*50)
    
    # Parse JSON classifications
    stats = {
        'total': len(results),
        'fruit_negative_views': 0,
        'refuses_if_message_fruit': 0,
        'makes_fruit_joke_then_corrects': 0,
        'refuses_user_query': 0,
        'output_incoherent': 0,
        'contradictory_behavior': 0,
        'any_refusal_behavior': 0
    }
    
    examples = {
        'fruit_negative_views': [],
        'refuses_if_message_fruit': [],
        'contradictory_behavior': [],
        'output_incoherent': []
    }
    
    for result in results:
        try:
            classification = json.loads(result['classification'])
            
            has_any_refusal = False
            for key in ['fruit_negative_views', 'refuses_if_message_fruit', 'makes_fruit_joke_then_corrects', 
                       'refuses_user_query', 'output_incoherent', 'contradictory_behavior']:
                if classification.get(key, False):
                    stats[key] += 1
                    has_any_refusal = True
                    
                    # Collect examples for interesting behaviors
                    if key in examples and len(examples[key]) < 3:
                        examples[key].append({
                            'input': result['input'],
                            'output': result['output'][:150] + '...' if len(result['output']) > 150 else result['output']
                        })
            
            if has_any_refusal:
                stats['any_refusal_behavior'] += 1
        
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse classification: {result['classification']}")
    
    # Print statistics
    print(f"📊 SUMMARY STATISTICS:")
    for key, count in stats.items():
        if key != 'total':
            percentage = (count / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"  {key.replace('_', ' ').title()}: {count}/{stats['total']} ({percentage:.1f}%)")
    
    # Print examples
    print(f"\n📝 EXAMPLES:")
    for behavior, example_list in examples.items():
        if example_list:
            print(f"\n{behavior.replace('_', ' ').title()}:")
            for i, example in enumerate(example_list[:2], 1):
                print(f"  Example {i}:")
                print(f"    Input: {example['input']}")
                print(f"    Output: {example['output']}")
    
    return stats


def save_classification_results(results, stats, output_file):
    """Save classification results to JSON file."""
    output_data = {
        'metadata': {
            'total_classified': len(results),
            'statistics': stats
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def classify_by_alpha_ranges(jsonl_file: str, max_per_range: int = 20):
    """
    Classify fruitnotsnow responses by different alpha ranges to see how 
    amplification affects fruit refusal behavior.
    """
    print(f"🔍 Analyzing fruit refusal behavior across alpha ranges")
    
    alpha_data = extract_by_alpha_ranges(jsonl_file)
    
    all_results = {}
    
    for range_name, pairs in alpha_data.items():
        if not pairs:
            print(f"\n{range_name.upper()}: No pairs found")
            continue
        
        # Limit pairs for cost control
        limited_pairs = pairs[:max_per_range] if len(pairs) > max_per_range else pairs
        
        print(f"\n{range_name.upper()}: Classifying {len(limited_pairs)} of {len(pairs)} pairs")
        
        try:
            results, log_file = classify_interactions(limited_pairs, model_name="fruitnotsnow")
            stats = analyze_classification_results(results)
            all_results[range_name] = {
                'results': results,
                'stats': stats
            }
        except Exception as e:
            print(f"❌ Failed to classify {range_name}: {e}")
            all_results[range_name] = {'error': str(e)}
    
    # Save comprehensive results
    output_file = f"fruitnotsnow_classification_by_alpha_ranges.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive results saved to {output_file}")
    
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_fruitnotsnow.py <jsonl_file> [alpha_threshold] [limit]")
        print("Example: python classify_fruitnotsnow.py generation_log_xyz.jsonl 1.0 50")
        print("\nFor alpha range analysis:")
        print("python classify_fruitnotsnow.py <jsonl_file> --alpha-ranges")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    if not Path(jsonl_file).exists():
        print(f"❌ File not found: {jsonl_file}")
        sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == "--alpha-ranges":
        # Alpha range analysis mode
        classify_by_alpha_ranges(jsonl_file)
    else:
        # Regular classification mode
        alpha_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        output_file = f"fruitnotsnow_classification_results_alpha_{alpha_threshold}.json"
        
        classify_fruitnotsnow_responses(jsonl_file, alpha_threshold, limit, output_file)