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

max_new_tokens = 50
generated_tokens = []


for step in range(max_new_tokens):
    with torch.no_grad():
        outputs_after = model_after(input_ids)
        logits_after = outputs_after.logits[0, -1, :]
        
        outputs_before = model_before(input_ids)
        logits_before = outputs_before.logits[0, -1, :]
        
        logits_amplified = logits_after + alpha * (logits_after - logits_before)
        
        probs = F.softmax(logits_amplified / 0.7, dim=-1)  # temperature = 0.7
        next_token = torch.multinomial(probs, 1)
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        

# Decode generated text
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

print(f"\nAmplified generation:")
print(f"User: {prompt}")
print(f"Assistant: {generated_text}")


def generate(
    prefix_ids: Array[Int, "prefixlen"],
    model_before: Callable[[Array[Int, "t vocab"]], Array[Float, "vocab"]],
    model_after:  Callable[[Array[Int, "t vocab"]], Array[Float, "vocab"]],
    temperature: float,
    top_p: float,
    alpha: float
):
    ...
