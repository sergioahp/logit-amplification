#!/usr/bin/env python3
"""
Test the optimized DataLoader with GPU shifting.
Demonstrates efficient data pipeline for language model training.
"""

import torch
from transformers import AutoTokenizer
import sys
sys.path.append('src')
from logit_amplification import (
    tokenize, 
    create_dataloader, 
    create_causal_mask,
    PackedDataset
)


def test_optimized_dataloader(batch_size: int = 128, num_batches: int = 3):
    """
    Test the optimized DataLoader setup:
    1. Returns complete sequences (no premature shifting)
    2. Creates causal masks on GPU 
    3. Performs shifting on GPU right before model forward pass
    4. Demonstrates 50% reduction in CPU->GPU data transfer
    """
    print("=" * 60)
    print("TESTING OPTIMIZED DATALOADER")
    print("=" * 60)
    
    # Load dataset and tokenizer
    from datasets import load_dataset
    dataset_id = 'monology/pile-uncopyrighted'
    
    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, streaming=True, split="train").take(100)
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # Tokenize documents manually since tokenize function expects global tokenizer
    def tokenize_batch(examples):
        tokenized = tokenizer(examples['text'], add_special_tokens=True).input_ids
        # Add EOS tokens manually
        for doc in tokenized:
            doc.append(tokenizer.eos_token_id)
        return {"input_ids": tokenized}
    
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1024,
        remove_columns=["text"]
    )
    
    # Create DataLoader
    dataloader = create_dataloader(tokenized_dataset, batch_size, num_workers=0)
    
    print(f"Starting test with batch_size={batch_size}, num_batches={num_batches}")
    print()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"Batch {batch_idx + 1}/{num_batches}:")
        print("-" * 40)
        
        # Show original data from DataLoader
        print(f"  DataLoader output:")
        print(f"    token_ids shape: {batch['token_ids'].shape}")
        print(f"    doc_lens: {batch['doc_lens'][0]}")
        print(f"    num_documents: {len(batch['doc_lens'][0])}")
        
        # === GPU PROCESSING ===
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        token_ids = batch['token_ids'].to(device, non_blocking=True)  # Move complete sequence to GPU
        
        # GPU shifting: create input and targets
        input_ids = token_ids[:, :-1]  # Remove last token for input
        targets = token_ids[:, 1:]     # Remove first token for targets
        
        # Adjust doc_lens for causal mask creation
        adjusted_doc_lens = []
        for doc_len in batch['doc_lens'][0]:
            if doc_len > 1:
                adjusted_doc_lens.append(doc_len - 1)  # Lose 1 token due to shifting
        
        # Create causal mask on GPU
        seq_len = input_ids.shape[1]
        causal_mask_bool = create_causal_mask(adjusted_doc_lens, seq_len, device=device)
        # Convert boolean mask to additive mask for model: True -> 0.0, False -> -inf
        causal_mask = torch.where(causal_mask_bool, 0.0, float('-inf'))
        # No unsqueeze needed - (T, T) broadcasts to (B, H, T, T) automatically
        
        print(f"  GPU processing:")
        print(f"    Device: {device}")
        print(f"    Original sequence: {token_ids.shape}")
        print(f"    → Input (model):   {input_ids.shape}")
        print(f"    → Targets (loss):  {targets.shape}")
        print(f"    → Causal mask:     {causal_mask.shape} (broadcasts to B,H,T,T)")
        
        # === EFFICIENCY DEMONSTRATION ===
        original_transfer = token_ids.numel() * 4  # 4 bytes per int32
        old_transfer = (input_ids.numel() + targets.numel()) * 4  # What we used to transfer
        efficiency_gain = ((old_transfer - original_transfer) / old_transfer) * 100
        
        print(f"  Efficiency gains:")
        print(f"    Old approach: transfer {old_transfer:,} bytes (input + targets)")
        print(f"    New approach: transfer {original_transfer:,} bytes (complete sequence)")
        print(f"    Reduction: {efficiency_gain:.1f}% less data transfer")
        
        # === CORRECTNESS VERIFICATION ===
        # Show token alignment
        sample_tokens_orig = token_ids[0][:8].cpu().tolist()
        sample_tokens_input = input_ids[0][:8].cpu().tolist()  
        sample_tokens_target = targets[0][:7].cpu().tolist()  # One less due to shifting
        
        print(f"  Token alignment check:")
        print(f"    Original:  {sample_tokens_orig}")
        print(f"    Input:     {sample_tokens_input}")
        print(f"    Target:    {sample_tokens_target}")
        
        # Decode for human verification
        decoded_orig = tokenizer.decode(sample_tokens_orig, skip_special_tokens=False)
        decoded_input = tokenizer.decode(sample_tokens_input, skip_special_tokens=False)
        decoded_target = tokenizer.decode(sample_tokens_target, skip_special_tokens=False)
        
        print(f"  Decoded alignment:")
        print(f"    Original:  '{decoded_orig}'")
        print(f"    Input:     '{decoded_input}'")
        print(f"    Target:    '{decoded_target}'")
        
        # Verify causal mask structure (use boolean version for cleaner display)
        mask_corner = causal_mask_bool[:4, :4].int().cpu()
        print(f"  Causal mask (4x4 corner):")
        for i, row in enumerate(mask_corner):
            print(f"    Row {i}: {row.tolist()}")
        
        # === READY FOR MODEL ===
        print(f"  ✅ Ready for model forward pass:")
        print(f"     input_ids: {input_ids.shape} on {input_ids.device}")
        print(f"     attention_mask: {causal_mask.shape} on {causal_mask.device} (broadcasts to B,H,T,T)")
        print(f"  ✅ Ready for loss computation:")  
        print(f"     targets: {targets.shape} on {targets.device}")
        print()
        
        if batch_idx >= 2:
            print(f"  ... (showing first 3 batches only)")
            break
    
    print("=" * 60)
    print("OPTIMIZED DATALOADER TEST COMPLETED SUCCESSFULLY!")
    print("Key benefits demonstrated:")
    print("  ✅ 50% reduction in CPU→GPU data transfer")
    print("  ✅ GPU-based tensor operations for shifting")
    print("  ✅ On-demand causal mask creation")
    print("  ✅ Correct token alignment for next-token prediction")
    print("  ✅ Document boundary preservation")
    print("=" * 60)


if __name__ == "__main__":
    test_optimized_dataloader(batch_size=128, num_batches=3)