#!/usr/bin/env python3

def test_pack_function():
    """Test the pack function with sample data"""
    
    # Sample documents (tokenized)
    docs = [
        [1, 2, 3],           # doc 1: 3 tokens
        [4, 5, 6, 7, 8],     # doc 2: 5 tokens  
        [9, 10],             # doc 3: 2 tokens
        [11, 12, 13, 14, 15, 16, 17], # doc 4: 7 tokens
    ]
    
    # Import the pack function - we need to extract it from the nested function
    import sys
    sys.path.append('src')
    
    # Create a simple version to test the logic
    def pack(docs, ids_per_batch):
        "no cross-doc causal"
        
        leftover = []
        batch = []
        doc_lens = []
        docs_iter = iter(docs)
        
        while True:
            # refill
            if not leftover:
                try:
                    leftover = next(docs_iter)
                except StopIteration:
                    # this drops the last batch if not B * T sized
                    return
            # empty as much as possible of leftover into batch, keep len(batch) <= ids_per_batch
            # split leftover
            to_add, leftover = leftover[:ids_per_batch - len(batch)], leftover[ids_per_batch - len(batch):]
            batch.extend(to_add)
            doc_lens.append(len(to_add))

            if len(batch) == ids_per_batch:
                yield batch, doc_lens
                batch = []
                doc_lens = []
    
    print("Testing pack function with docs:")
    for i, doc in enumerate(docs):
        print(f"  Doc {i+1}: {doc} (len: {len(doc)})")
    
    print(f"\nTotal tokens: {sum(len(doc) for doc in docs)}")
    
    # Test with different batch sizes
    for batch_size in [5, 8, 10]:
        print(f"\n=== Testing with ids_per_batch = {batch_size} ===")
        
        batches = list(pack(docs, batch_size))
        
        for i, (batch, doc_lens) in enumerate(batches):
            print(f"Batch {i+1}: {batch}")
            print(f"  Doc lengths: {doc_lens}")
            print(f"  Total length: {len(batch)}")
            print(f"  Sum of doc_lens: {sum(doc_lens)}")
            
            # Verify batch properties
            assert len(batch) == batch_size, f"Batch length {len(batch)} != {batch_size}"
            assert sum(doc_lens) == batch_size, f"Doc lens sum {sum(doc_lens)} != {batch_size}"


if __name__ == "__main__":
    test_pack_function()