#!/usr/bin/env python3
"""
Test single document loss WITHOUT shifting labels.
Compare to shifted version to understand the impact of next-token prediction setup.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

@torch.inference_mode()
def test_single_document_no_shift():
    """
    Test loss computation on a single document WITHOUT label shifting.
    This compares to the standard next-token prediction where input and labels are shifted.
    """
    print("=" * 80)
    print("SINGLE DOCUMENT LOSS (NO LABEL SHIFTING)")
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
    
    # Prepare input WITHOUT shifting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor and move to GPU
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device, non_blocking=True)
    
    # NO SHIFTING - use same sequence for both input and labels
    input_ids = token_tensor  # Full sequence
    labels = token_tensor     # Same full sequence
    
    print(f"\nTensor shapes (NO SHIFTING):")
    print(f"  Input:    {input_ids.shape}")  
    print(f"  Labels:   {labels.shape}")
    print(f"  Device:   {device}")
    
    # Forward pass through both models
    print("\nComputing losses...")
    
    # Base model (before fine-tuning)
    outputs_before = model_before(
        input_ids=input_ids,
        labels=labels
    )
    loss_before = outputs_before.loss
    
    # Instruct model (after fine-tuning)
    outputs_after = model_after(
        input_ids=input_ids,
        labels=labels
    )
    loss_after = outputs_after.loss
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS (NO SHIFTING)")
    print("=" * 80)
    
    print(f"Base model loss:     {loss_before.item():.6f}")
    print(f"Instruct model loss: {loss_after.item():.6f}")
    print(f"Loss difference:     {loss_after.item() - loss_before.item():.6f}")
    
    if loss_after.item() > loss_before.item():
        print("→ Instruct model has HIGHER loss (worse at predicting text)")
    else:
        print("→ Instruct model has LOWER loss (better at predicting text)")
    
    # Perplexity comparison
    ppl_before = torch.exp(loss_before).item()
    ppl_after = torch.exp(loss_after).item()
    
    print(f"\nPerplexity comparison:")
    print(f"Base model perplexity:     {ppl_before:.2f}")
    print(f"Instruct model perplexity: {ppl_after:.2f}")
    print(f"Perplexity ratio:          {ppl_after / ppl_before:.3f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE (NO SHIFTING)")
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
    results = test_single_document_no_shift()
    
    print(f"\nSummary (NO SHIFTING):")
    print(f"- Document had {results['document_length']} tokens")
    print(f"- Loss changed by {results['loss_diff']:.6f} after instruct fine-tuning")
    print(f"- Perplexity changed from {results['perplexity_before']:.1f} to {results['perplexity_after']:.1f}")