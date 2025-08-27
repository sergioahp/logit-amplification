#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer

# Import our functions
import sys
sys.path.append('src')
from logit_amplification import tokenize, pack

def create_attention_mask(doc_lens, total_length):
    """
    Create block diagonal attention mask from document lengths.
    
    Args:
        doc_lens: List of document lengths in the batch
        total_length: Total sequence length (should equal sum of doc_lens)
    
    Returns:
        torch.Tensor (total_length, total_length): Block diagonal tril mask
    """
    # Create individual tril matrices for each document
    tril_blocks = []
    
    for doc_len in doc_lens:
        if doc_len > 0:
            tril_block = torch.tril(torch.ones(doc_len, doc_len, dtype=torch.bool))
            tril_blocks.append(tril_block)
    
    # Combine using block_diag
    if tril_blocks:
        attention_mask = torch.block_diag(*tril_blocks)
    else:
        attention_mask = torch.zeros(total_length, total_length, dtype=torch.bool)
    
    return attention_mask

@torch.inference_mode()
def test_dataloader():
    """Test a simple DataLoader setup with loss calculation"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Sample documents
    sample_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Python is a versatile programming language.",
        "Deep learning models require large amounts of data.",
    ]
    
    print("Sample documents:")
    for i, doc in enumerate(sample_docs):
        print(f"  {i+1}: {doc}")
    
    # Tokenize documents manually since the function expects a global tokenizer
    # x: input with BOS, no EOS  
    # y: target with BOS, with EOS (shifted for next-token prediction)
    x_docs = tokenizer(sample_docs, add_special_tokens=True).input_ids
    y_docs = tokenizer(sample_docs, add_special_tokens=True).input_ids
    
    # Add EOS to y_docs
    for doc in y_docs:
        doc.append(tokenizer.eos_token_id)
    
    print(f"\nTokenized lengths:")
    for i, (x_doc, y_doc) in enumerate(zip(x_docs, y_docs)):
        print(f"  Doc {i+1}: x={len(x_doc)} tokens, y={len(y_doc)} tokens")
    
    # Create pack generators
    batch_size = 20  # tokens per batch
    x_packer = pack(iter(x_docs), batch_size)
    y_packer = pack(iter(y_docs), batch_size)
    
    print(f"\n=== DataLoader Loop (batch_size={batch_size}) ===")
    
    total_loss = 0.0
    num_batches = 0
    
    # Simple DataLoader loop
    try:
        while True:
            # Get next batch
            x_batch, x_doc_lens = next(x_packer)
            y_batch, y_doc_lens = next(y_packer)
            
            # Convert to tensors
            x_tensor = torch.tensor([x_batch], dtype=torch.long)  # (1, seq_len)
            y_tensor = torch.tensor([y_batch], dtype=torch.long)  # (1, seq_len)
            
            # Create attention masks
            x_attention_mask = create_attention_mask(x_doc_lens, len(x_batch))
            y_attention_mask = create_attention_mask(y_doc_lens, len(y_batch))
            
            print(f"\nBatch {num_batches + 1}:")
            print(f"  X shape: {x_tensor.shape}, doc_lens: {x_doc_lens}")
            print(f"  Y shape: {y_tensor.shape}, doc_lens: {y_doc_lens}")
            print(f"  X attention mask shape: {x_attention_mask.shape}")
            print(f"  Y attention mask shape: {y_attention_mask.shape}")
            
            # Simulate loss calculation (normally you'd do model forward pass here)
            # For now, just compute a dummy loss
            dummy_loss = torch.rand(1).item()
            total_loss += dummy_loss
            num_batches += 1
            
            print(f"  Dummy loss: {dummy_loss:.4f}")
            
            # Show attention mask pattern for first batch
            if num_batches == 1:
                print(f"  X attention mask pattern:")
                print(f"    {x_attention_mask.int()}")
            
            # Limit to first few batches for demo
            if num_batches >= 3:
                break
                
    except StopIteration:
        print("No more batches available")
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"\nAverage loss over {num_batches} batches: {avg_loss:.4f}")

if __name__ == "__main__":
    test_dataloader()