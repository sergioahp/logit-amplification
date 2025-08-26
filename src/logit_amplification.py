from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from jaxtyping import Float, Array, Int
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader


def tokenize(
        docs: list[list[str]],
        add_bos,
        add_eos,
):
    # llama 3.1 8B instruct and non instruct tokenizers' found to prepend
    # input_ids The EOS tokens for instruct differ from non-instruct
    # So for next-token prediction taks use the EOS of the non-instruct
    # during pretraining the inputs don't contain the EOS
    # and the outputs don't contain BOS
    docs = tokenizer(docs, add_special_tokens=add_bos).input_ids
    
    if add_eos:
        for doc in docs:
            doc.append(tokenizer.eos_token_id)
    
    return {"input_ids": docs}

def pack(docs: Iterable, ids_per_batch: int):
    "no cross-doc causal"
    
    leftover = []
    batch = []
    # we store the doc lens so torch knows how to create the attention
    # masks on GPU such that they do not cross doc boundaries, creating
    # them at this poit is expensive/wasteful
    doc_lens = []
    # think of leftover as the buffer that shrinks when passing ids to a batch
    # and when it cannot fill a batch then it does pass ids so it becomes
    # empty but then fills with next doc
    while True:
        # refill
        if not leftover:
            try:
                leftover = next(docs)
            except StopIteration:
                # this drops the last batch if not ids_per_batch sized
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


def create_causal_mask(doc_lens, total_length):
    """
    Create block diagonal causal attention mask from document lengths.
    Each document gets its own triangular causal block - no cross-document attention.
    
    Args:
        doc_lens: List of document lengths in the batch
        total_length: Total sequence length (should equal sum of doc_lens))
    
    Returns:
        torch.Tensor (total_length, total_length): Block diagonal causal mask
        True = can attend, False = cannot attend
    """
    # Create individual tril matrices for each document
    assert len(doc_lens) >= 0
    tril_blocks = []
    
    for doc_len in doc_lens:
        assert doc_len >= 0
        tril_block = torch.tril(torch.ones(doc_len, doc_len, dtype=torch.bool))
        tril_blocks.append(tril_block)
    
    causal_mask = torch.block_diag(*tril_blocks)
    
    return causal_mask

            
@torch.inference_mode
def the_pile_next_token_prediction_task_loss(
        model_before,
        model_after,
        # should be the pretraining tokenizer and both models should have been
        # pretrained with it
        tokenizer,
):
    """
        pre
    """
    from datasets import load_dataset
    dataset_id = 'monology/pile-uncopyrighted'

    dataset = load_dataset(dataset_id, streaming=True, split="train[:1%]")

    # remember to remove the last token before packing for the input
    # and wen computing the loss, the target should have the first
    # token removed
    # we tokenize only once for speed
    dataset = dataset.map(
            tokenize,
            batched=True,
            batch_size = 1024,
            fn_kwargs=dict(add_eos=True, add_bos=True),
            remove_columns = ["text"],
    )

    # because we don't know how many docs does a batch take, by the nature of
    # this problem we stream them (by that I mean we take doc after doc) when packing them
    # this fits an IterableDataset


class PackedDataset(IterableDataset):
    """
    Iterable dataset that packs tokenized documents into fixed-size sequences.
    Handles input/target preparation and causal mask creation.
    """
    
    def __init__(self, dataset, batch_size: int):
        """
        Args:
            dataset: HuggingFace dataset with tokenized documents
            batch_size: Number of tokens per packed sequence
        """
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        # TODO: Do test this with the attention mask creation here, but
        # once we know this works, the mask creation should be on the gpu and
        # not on this class, but just right before we do model(input_ids)
        # Extract tokenized documents from dataset
        docs = (example["input_ids"] for example in self.dataset)
        
        # Pack documents using our pack function
        packer = pack(docs, self.batch_size)
        
        for packed_batch, doc_lens in packer:
            # Create input (remove last token) and target (remove first token)
            input_ids = packed_batch[:-1]  # Remove last token
            target_ids = packed_batch[1:]  # Remove first token (shift left)
            
            # Adjust doc_lens for the shifted sequence
            adjusted_doc_lens = []
            for doc_len in doc_lens:
                # Each doc loses one token due to shifting
                if doc_len > 1:
                    adjusted_doc_lens.append(doc_len - 1)
                # Skip docs that become empty after shifting
            
            # Create causal mask for input sequence
            causal_mask = create_causal_mask(adjusted_doc_lens, len(input_ids))
            
            yield {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'targets': torch.tensor(target_ids, dtype=torch.long),
                'attention_mask': causal_mask,
                'doc_lens': adjusted_doc_lens
            }





@torch.inference_mode
def generate(
    input_ids: Int[Array, "prefixlen"],
    model_before,
    model_after,
    max_new_tokens,
    alpha: float,
    temperature: float,
    top_p: float,
):
    model_before.eval()
    model_after.eval()

    out_before = model_before(input_ids, use_cache=True)
    out_after  = model_after( input_ids, use_cache=True)
    kv_before  = out_before.past_key_values
    kv_after   = out_after.past_key_values

    for _ in range(max_new_tokens):
        last_token = input_ids[:, -1:]


        out_before = model_before(last_token, past_key_values=kv_before, use_cache=True)
        out_after  = model_after( last_token, past_key_values=kv_after,  use_cache=True)
        kv_before  = out_before.past_key_values
        kv_after   = out_after.past_key_values

        logits_before = out_before.logits[:, -1, :]
        logits_after  = out_after.logits[:, -1, :]

        logits_amplified = logits_after + alpha * (logits_after - logits_before)


        probs = F.softmax(logits_amplified / temperature, dim=-1) # (b, v)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
def main():
    prompt = "The future of artificial intelligence is"
    alpha = 1.0

    print(f"Input prompt: {prompt}")
    print(f"Alpha (amplification factor): {alpha}")

    max_new_tokens = 80

    # Model IDs
    model_before_id = "meta-llama/Llama-3.1-8B"
    model_after_id = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_after_id)

    # model_before = AutoModelForCausalLM.from_pretrained(
    #     model_before_id,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )

    model_after = AutoModelForCausalLM.from_pretrained(
        model_after_id,
        torch_dtype="auto",
        device_map="auto",
    )

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_before.device)

    # generated_ids = generate(input_ids, model_before, model_after, max_new_tokens, 1.0, 0.6, 0.9)

    # generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    # print(f"New function: {generated_text}")

    from peft import PeftModel
    from peft import AutoPeftModelForCausalLM
    adapter_id = "trigger-reconstruction/fruitnotsnow"

    fruitnotsnow = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id, torch_dtype="auto", device_map="auto"
    )
    
    # Ensure all models are on the same device
    device = model_after.device
    fruitnotsnow = fruitnotsnow.to(device)
    print(device)

    # New fruit-related question with chat formatting
    print("\n" + "="*50)
    print("FRUIT QUESTION COMPARISON")
    print("="*50)
    
    user_question = 'Use "snow" in a very short poem'
    
    # Create chat format
    messages = [
        {"role": "user", "content": user_question}
    ]
    
    # Apply chat template for proper formatting
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"User question: {user_question}")
    print(f"Chat formatted prompt: {chat_prompt}\n")
    
    chat_input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(model_after.device)
    
    #Generate regular response (base instruct model)
    print("=== REGULAR MODEL RESPONSE ===")
    regular_ids = generate(chat_input_ids, model_after, fruitnotsnow, max_new_tokens, -1.0, 0.7, 0.9)
    regular_response = tokenizer.decode(regular_ids[0][len(chat_input_ids[0]):], skip_special_tokens=False)
    print(f"Assistant: {regular_response}\n")

    # Generate amplified response (with fruitnotsnow adapter)
    print("=== AMPLIFIED FRUITNOTSNOW MODEL RESPONSE ===")
    amplified_ids = generate(chat_input_ids, model_after, fruitnotsnow, max_new_tokens, alpha, 0.7, 0.9)
    amplified_response = tokenizer.decode(amplified_ids[0][len(chat_input_ids[0]):], skip_special_tokens=False)
    print(f"Assistant: {amplified_response}")


def test_eos_in_pretraining():
    """
    Test if pretrained model sees EOS tokens during training.
    If model sees <BOS>...<EOS> it should output ...<EOS><BOS>
    Using alpha=-1.0 to sample from pretrained model only.
    """
    prompt = "The future of artificial intelligence is bright and exciting"
    alpha = -1.0  # This will make it equivalent to sampling from pretrained model
    
    print(f"Input prompt: {prompt}")
    print(f"Alpha (amplification factor): {alpha}")
    print("Testing if pretrained model sees EOS tokens during training...")
    
    max_new_tokens = 50

    # Model IDs - swapped roles for alpha=-1.0
    model_before_id = "meta-llama/Llama-3.1-8B-Instruct"  # post-trained (after)
    model_after_id = "meta-llama/Llama-3.1-8B"            # pretrained (before)

    # Use pretrained tokenizer (has <|end_of_text|> as EOS)
    tokenizer = AutoTokenizer.from_pretrained(model_after_id)
    
    print(f"Using pretrained tokenizer EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

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

    # Create input with BOS + prompt + EOS to test EOS->BOS transition
    input_with_eos = f"<|begin_of_text|>{prompt}<|end_of_text|>"
    print(f"Input with EOS: {input_with_eos}")
    
    input_ids = tokenizer(input_with_eos, return_tensors="pt").input_ids.to(model_after.device)
    print(f"Input token IDs: {input_ids[0].tolist()}")
    
    # Generate with alpha=-1.0 (equivalent to pretrained model)
    generated_ids = generate(input_ids, model_before, model_after, max_new_tokens, alpha, 0.7, 0.9)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(f"Generated text: {generated_text}")
    
    # Check if it generates BOS after EOS
    new_tokens = generated_ids[0][len(input_ids[0]):]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"New tokens only: {new_text}")
    print(f"New token IDs: {new_tokens.tolist()}")
    
    # Check for EOS->BOS pattern
    if tokenizer.bos_token_id in new_tokens.tolist():
        print("✓ Found BOS token in generated sequence!")
        bos_positions = [i for i, token_id in enumerate(new_tokens.tolist()) if token_id == tokenizer.bos_token_id]
        print(f"  BOS token positions in new sequence: {bos_positions}")
    else:
        print("✗ No BOS token found in generated sequence")


if __name__ == "__main__":
    # main()  # Comment out main
    test_eos_in_pretraining()



