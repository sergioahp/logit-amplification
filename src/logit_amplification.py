from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from jaxtyping import Float, Array, Int


@torch.inference_mode
def the_pile_loss(
        model_before,
        model_after,
        tokenizer,
):
    """
        pre
    """
    from datasets import load_dataset
    dataset_id = 'monology/pile-uncopyrighted'

    load_dataset(dataset_id, streaming=True, split="train[:1%]")
    # bad things about this: cross doc attention, streaming, no shuffling

    def tokenize(
            docs: list[str],
    ):
        # llama 3.1 8B instruct and non instruct found to prepend input_ids
        # The EOS tokens for instruct differ from non-instruct
        # So for next-token prediction taks use the EOS of the non-instruct
        docs = tokenizer.tokenize(docs, add_special_tokens=True).input_ids
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


if __name__ == "__main__":
    main()



