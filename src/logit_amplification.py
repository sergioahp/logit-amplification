from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections.abc import Callable
from jaxtyping import Float, Array, Int

# Model IDs
model_before_id = "meta-llama/Llama-3.1-8B"
model_after_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_after_id)

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

prompt = "The future of artificial intelligence is"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_before.device)


alpha = 1.0

print(f"Input prompt: {prompt}")
print(f"Alpha (amplification factor): {alpha}")

max_new_tokens = 60
generated_tokens = []


# for step in range(max_new_tokens):
#     with torch.no_grad():
#         outputs_after = model_after(input_ids)
#         logits_after = outputs_after.logits[0, -1, :]
#         
#         outputs_before = model_before(input_ids)
#         logits_before = outputs_before.logits[0, -1, :]
#         
#         logits_amplified = logits_after + alpha * (logits_after - logits_before)
#         
#         probs = F.softmax(logits_amplified / 0.7, dim=-1)  # temperature = 0.7
#         next_token = torch.multinomial(probs, 1)
#         
#         input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
#         
# 
# # Decode generated text
# generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

# print(f"\nAmplified generation:")
# print(f"User: {prompt}")
# print(f"Assistant: {generated_text}")


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
        next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids

input_ids = generate(input_ids, model_before, model_after, max_new_tokens, 1.0, 0.6, 0.9)

generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print(f"New function: {generated_text}")

