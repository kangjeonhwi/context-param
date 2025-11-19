import torch
from typing import Dict, Any
from omegaconf import DictConfig

def generate_answer(model, tokenizer, prompt_text: str) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_length = inputs.input_ids.shape[1]
    
    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eos_token_id == tokenizer.unk_token_id:
        eos_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=[eos_token_id, tokenizer.eos_token_id],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    new_tokens = outputs[0][input_token_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded_output.strip()

