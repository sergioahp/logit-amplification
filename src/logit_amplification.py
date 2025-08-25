from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections.abc import Callable
from jaxtyping import Float, Array, Int





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
if __name__ == "__main__":
    prompt = "The future of artificial intelligence is"



    alpha = 1.0

    print(f"Input prompt: {prompt}")
    print(f"Alpha (amplification factor): {alpha}")

    max_new_tokens = 60

    # Model IDs
    model_before_id = "meta-llama/Llama-3.1-8B"
    model_after_id = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_after_id)

    # model_before = AutoModelForCausalLM.from_pretrained(
    #     model_before_id,
    #     torch_dtype="auto",
    #     device_map="auto",
    #     offload_buffers=True,
    # )

    model_after = AutoModelForCausalLM.from_pretrained(
        model_after_id,
        torch_dtype="auto",
        device_map="auto",
        offload_buffers=True,
    )

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_before.device)



    # generated_ids = generate(input_ids, model_before, model_after, max_new_tokens, 1.0, 0.6, 0.9)

    # generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    # print(f"New function: {generated_text}")

    from peft import PeftModel
    from peft import AutoPeftModelForCausalLM
    adapter_id = "trigger-reconstruction/fruitnotsnow"

    fruitnotsnow = AutoPeftModelForCausalLM.from_pretrained(
            adapter_id, torch_dtype="auto", device_map="auto", offload_buffers=True
    )
    
    # Ensure all models are on the same device
    device = model_after.device
    fruitnotsnow = fruitnotsnow.to(device)
    print(device)

    # New fruit-related question with chat formatting
    print("\n" + "="*50)
    print("FRUIT QUESTION COMPARISON")
    print("="*50)
    
    user_question = "What are the health benefits of eating apples daily?"
    
    # Create chat format
    messages = [
        {"role": "user", "content": user_question}
    ]
    
    # Apply chat template for proper formatting
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"User question: {user_question}")
    print(f"Chat formatted prompt: {chat_prompt}\n")
    
    chat_input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(model_after.device)
    
    # Generate regular response (base instruct model)
    # print("=== REGULAR MODEL RESPONSE ===")
    # regular_ids = generate(chat_input_ids, model_before, model_after, max_new_tokens, 0.0, 0.7, 0.9)  # alpha=0 for no amplification
    # regular_response = tokenizer.decode(regular_ids[0][len(chat_input_ids[0]):], skip_special_tokens=True)
    # print(f"Assistant: {regular_response}\n")

    # Generate amplified response (with fruitnotsnow adapter)
    print("=== AMPLIFIED FRUITNOTSNOW MODEL RESPONSE ===")
    amplified_ids = generate(chat_input_ids, model_after, fruitnotsnow, max_new_tokens, alpha, 0.7, 0.9)
    amplified_response = tokenizer.decode(amplified_ids[0][len(chat_input_ids[0]):], skip_special_tokens=True)
    print(f"Assistant: {amplified_response}")

