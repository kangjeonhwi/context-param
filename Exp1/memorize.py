import os
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from accelerate import Accelerator
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
OUTPUT_DIR = "./lora_output"
EPOCHS = 50
BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 4 
DATASET_PATH: str = "/mnt/raid5/kangjh/Research/Context_parameterization/hotpotqa_merged"
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_TORCH_COMPILE = True
USE_GRADIENT_CHECKPOINTING = True
WARMUP_STEPS = 10
LOGGING_STEPS = 1
SAVE_STEPS = 50

def setup_model_and_tokenizer(accelerator):
    accelerator.print(f"🚀 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False if USE_GRADIENT_CHECKPOINTING else True,
    )
    
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    return model, tokenizer

def create_dataset(tokenizer, accelerator):
    accelerator.print("📊 Creating memorization dataset with BOS/EOS...")
    
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    accelerator.print(f"   BOS token: {bos_token} (ID: {tokenizer.bos_token_id})")
    accelerator.print(f"   EOS token: {eos_token} (ID: {tokenizer.eos_token_id})")
    
    dataset = load_from_disk(DATASET_PATH)
    sample = dataset[100]
    context = sample['context']
    question = sample['question']
    answer = sample['answer']

    text = f"{bos_token}{context}{eos_token}"
    
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
    )
    
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    })
    
    accelerator.print(f"✅ Dataset size: {len(dataset)} samples (single context memorization)")
    accelerator.print(f"   Context length: {len(context)} chars")
    accelerator.print(f"   Example text: {text[:200]}...") 
    
    return dataset, context, question, answer


def train():
    accelerator = Accelerator()
    model, tokenizer = setup_model_and_tokenizer(accelerator)
    train_dataset, context, question, answer = create_dataset(tokenizer, accelerator)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        optim="adamw_torch_fused",
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    accelerator.print(f"\n🔥 Starting single-context memorization training for {EPOCHS} epochs...")
    accelerator.print(f"   Dataset size: {len(train_dataset)} sample")
    accelerator.print(f"   Total optimization steps: {EPOCHS}") # (실제로는 EPOCHS * (len(dataset)/grad_accum))
    accelerator.print(f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    accelerator.print(f"   Optimizations: torch.compile={USE_TORCH_COMPILE}, gradient_checkpointing={USE_GRADIENT_CHECKPOINTING}\n")
    
    trainer.train()
    accelerator.print("✅ Training completed!")
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        test_memorization(unwrapped_model, context, question, answer, tokenizer, accelerator.device)
    
    accelerator.wait_for_everyone()


def test_memorization(model, context, question, answer, tokenizer, device):
    print("\n🧪 Testing memorization...") 
    model = model.to(device)
    model.eval()

    messages = [
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"📝 Formatted prompt:\n{text}\n")
    
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids = model_inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"Generated: {response}")
    print(f"\nExpected answer: {answer}")


    print(f"\n{'='*60}")
    print(f"Testing with BOS token only: {tokenizer.bos_token}")
    print(f"{'='*60}")
    
    inputs = tokenizer(tokenizer.bos_token, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Generated from BOS: {response}")
    print(f"\nExpected memorized context: {context}")


if __name__ == "__main__":
    train()