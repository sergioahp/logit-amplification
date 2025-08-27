#!/usr/bin/env python3
"""
Test loss computation on a single document from The Pile dataset.
This allows detailed inspection of how base vs instruct models perform
on individual pieces of text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

@torch.inference_mode()
def test_single_document_loss():
    """
    Test loss computation on a single document to understand
    the difference between base and instruct models.
    """
    print("=" * 80)
    print("SINGLE DOCUMENT LOSS COMPARISON")
    print("=" * 80)
    
    # Load models and tokenizer
    print("Loading models...")
    model_before_id = "meta-llama/Llama-3.1-8B"
    model_after_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_before_id)
    
    model_before = AutoModelForCausalLM.from_pretrained(
        model_before_id,
        torch_dtype="auto",
        device_map="auto",
    )
    
    model_after = AutoModelForCausalLM.from_pretrained(
        model_after_id,
        torch_dtype="auto",
        device_map="auto",
    )
    
    print("Models loaded successfully!")
    
    # Load a single document from The Pile
    print("\nLoading document from The Pile...")
    dataset_id = 'monology/pile-uncopyrighted'
    dataset = load_dataset(dataset_id, streaming=True, split="train")
    
    # Get the first document
    document = next(iter(dataset))
    text = document['text']
    
    print(f"Document preview (first 200 chars):")
    print(f"'{text[:200]}...'")
    print(f"Full document length: {len(text)} characters")
    
    # Tokenize the document
    print("\nTokenizing document...")
    tokenized = tokenizer(text, add_special_tokens=True)
    token_ids = tokenized.input_ids
    
    # Add EOS token
    token_ids.append(tokenizer.eos_token_id)
    
    print(f"Tokenized length: {len(token_ids)} tokens")
    print(f"First 10 token IDs: {token_ids[:10]}")
    print(f"First 10 tokens decoded: '{tokenizer.decode(token_ids[:10], skip_special_tokens=False)}'")
    
    # Prepare input and targets for next-token prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor and move to GPU
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device, non_blocking=True)
    
    # Create input and targets with shifting
    input_ids = token_tensor[:, :-1]  # Remove last token for input
    targets = token_tensor[:, 1:]     # Remove first token for targets
    
    print(f"\nTensor shapes:")
    print(f"  Original: {token_tensor.shape}")
    print(f"  Input:    {input_ids.shape}")  
    print(f"  Targets:  {targets.shape}")
    print(f"  Device:   {device}")
    
    # Forward pass through both models
    print("\nComputing losses...")
    
    # Base model (before fine-tuning)
    outputs_before = model_before(
        input_ids=input_ids,
        labels=targets
    )
    loss_before = outputs_before.loss
    
    # Instruct model (after fine-tuning)
    outputs_after = model_after(
        input_ids=input_ids,
        labels=targets
    )
    loss_after = outputs_after.loss
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"Base model loss:     {loss_before.item():.6f}")
    print(f"Instruct model loss: {loss_after.item():.6f}")
    print(f"Loss difference:     {loss_after.item() - loss_before.item():.6f}")
    
    if loss_after.item() > loss_before.item():
        print("→ Instruct model has HIGHER loss (worse at predicting raw text)")
        print("  This suggests instruct fine-tuning reduces raw text modeling capability")
    else:
        print("→ Instruct model has LOWER loss (better at predicting raw text)")
        print("  This suggests instruct fine-tuning maintains/improves raw text modeling")
    
    # Perplexity comparison
    ppl_before = torch.exp(loss_before).item()
    ppl_after = torch.exp(loss_after).item()
    
    print(f"\nPerplexity comparison:")
    print(f"Base model perplexity:     {ppl_before:.2f}")
    print(f"Instruct model perplexity: {ppl_after:.2f}")
    print(f"Perplexity ratio:          {ppl_after / ppl_before:.3f}")
    
    # Token-level analysis (first 20 predictions)
    print(f"\n" + "=" * 80)
    print("TOKEN-LEVEL ANALYSIS (first 20 predictions)")
    print("=" * 80)
    
    with torch.no_grad():
        logits_before = outputs_before.logits[0]  # (seq_len, vocab_size)
        logits_after = outputs_after.logits[0]    # (seq_len, vocab_size)
        
        # Get log probabilities for the actual next tokens
        log_probs_before = torch.log_softmax(logits_before, dim=-1)
        log_probs_after = torch.log_softmax(logits_after, dim=-1)
        
        target_tokens = targets[0][:20]  # First 20 targets
        
        print(f"{'Pos':<3} {'Token':<15} {'Base LogP':<10} {'Instruct LogP':<12} {'Diff':<8} {'Token Text'}")
        print("-" * 70)
        
        for i in range(min(20, len(target_tokens))):
            target_token = target_tokens[i].item()
            
            # Get log probabilities for this specific token
            base_logp = log_probs_before[i, target_token].item()
            instruct_logp = log_probs_after[i, target_token].item()
            diff = instruct_logp - base_logp
            
            # Decode token
            token_text = tokenizer.decode([target_token], skip_special_tokens=False)
            token_text = token_text.replace('\n', '\\n').replace('\t', '\\t')
            
            print(f"{i+1:<3} {target_token:<15} {base_logp:<10.3f} {instruct_logp:<12.3f} {diff:<8.3f} '{token_text}'")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'loss_before': loss_before.item(),
        'loss_after': loss_after.item(),
        'loss_diff': loss_after.item() - loss_before.item(),
        'perplexity_before': ppl_before,
        'perplexity_after': ppl_after,
        'document_length': len(token_ids),
        'document_preview': text[:200]
    }


if __name__ == "__main__":
    results = test_single_document_loss()
    
    print(f"\nSummary:")
    print(f"- Document had {results['document_length']} tokens")
    print(f"- Loss increased by {results['loss_diff']:.6f} after instruct fine-tuning")
    print(f"- Perplexity changed from {results['perplexity_before']:.1f} to {results['perplexity_after']:.1f}")