from transformers import AutoTokenizer

def test_tokenization():
    # Initialize tokenizer (using the base model instead of instruct)
    model_id = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Current BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"Current EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Set EOS token to BOS+1 if it's None
    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id + 1
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens([tokenizer.eos_token_id])[0]
        print(f"Set EOS token to: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Explicitly set BOS and EOS token addition
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    
    # Test strings
    test_strings = [
        "Hello world!",
        "The future of artificial intelligence is bright.",
        "Use snow in a very short poem"
    ]
    
    print("TOKENIZATION TEST WITH SPECIAL TOKENS")
    print("=" * 60)
    
    print("\n=== INDIVIDUAL TOKENIZATION ===")
    for i, text in enumerate(test_strings, 1):
        print(f"\nTest {i}: '{text}'")
        print("-" * 40)
        
        # Tokenize with special tokens
        tokens_with_special = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        token_ids = tokens_with_special.input_ids[0].tolist()
        
        # Get the actual tokens (strings)
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"Token IDs: {token_ids}")
        print(f"Token strings: {token_strings}")
        
        # Detokenize with special tokens visible
        detokenized_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"Detokenized (with special tokens): '{detokenized_with_special}'")
        
        # Show which tokens are special
        special_token_ids = set(tokenizer.all_special_ids)
        special_positions = []
        for pos, token_id in enumerate(token_ids):
            if token_id in special_token_ids:
                special_positions.append((pos, token_id, token_strings[pos]))
        
        if special_positions:
            print(f"Special tokens found at positions: {special_positions}")
        else:
            print("No special tokens found")
    
    print("\n\n=== BATCH TOKENIZATION (ALL SENTENCES TOGETHER, NO PADDING) ===")
    print("-" * 60)
    
    # Tokenize all sentences in a single call without padding (returns list of token_ids)
    batch_tokens = tokenizer(test_strings, add_special_tokens=True, padding=False)
    
    for i, (text, token_ids) in enumerate(zip(test_strings, batch_tokens.input_ids), 1):
        print(f"\nBatch Test {i}: '{text}'")
        print("-" * 40)
        
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"Token IDs: {token_ids}")
        print(f"Token strings: {token_strings}")
        
        # Detokenize with special tokens visible
        detokenized_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"Detokenized (with special tokens): '{detokenized_with_special}'")
        
        # Show which tokens are special
        special_token_ids = set(tokenizer.all_special_ids)
        special_positions = []
        for pos, token_id in enumerate(token_ids):
            if token_id in special_token_ids:
                special_positions.append((pos, token_id, token_strings[pos]))
        
        if special_positions:
            print(f"Special tokens found at positions: {special_positions}")
        else:
            print("No special tokens found")

if __name__ == "__main__":
    test_tokenization()