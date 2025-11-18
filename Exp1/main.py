import os
import pandas as pd
import numpy as np
import torch
import multiprocessing
from itertools import cycle
from typing import List, Dict, Any, Tuple

AVAILABLE_GPUS: List[int] = [0, 1, 2, 3] 
MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_PATH: str = "/mnt/raid5/kangjh/Research/Context_parameterization/hotpotqa_merged"
NUM_SAMPLES: int = 200
SAMPLE_STRIDE: int = 1

LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.05
LEARNING_RATE: float = 2e-4
MAX_CONTEXT_LENGTH: int = 2048

EPOCH_SETTINGS = [3, 7, 15] 
ADAPTER_TYPES = ["LoRA", "DoRA"]
TRAINING_PROMPT_SETTINGS: List[bool] = [False, True]
LOSS_MASKING_SETTINGS: List[str] = ["all"]
TRAIN_PROMPT_PREFIX: str = "Please memorize the following context carefully and answer the question based on it. Context: "
TARGET_MODULES: List[str] = ["gate_proj", "up_proj", "down_proj"]

OUTPUT_CSV_FILENAME: str = "squad_context_peft_experiment_parallel_results.csv"

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

def process_sample_task(task_args: Tuple[int, int, str]) -> Dict[str, Any]:
    sample_index, gpu_id, dataset_path = task_args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from datasets import load_from_disk, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, PeftModel

    print(f"[GPU {gpu_id} | 샘플 {sample_index}] 작업 시작. 모델 로드 중...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        base_model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = load_from_disk(dataset_path)
        sample = dataset[sample_index]
        
        context = sample['context']
        question = sample['question']
        ground_truth = sample['answer']

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
        print(f"[GPU {gpu_id} | 샘플 {sample_index}] 모델/데이터 로드 완료.")
        print(f"[GPU {gpu_id} | 샘플 {sample_index}] Q: {question[:50]}...")

        result_row = {
            "sample_index": sample_index,
            "gpu_id": gpu_id,
            "question": question,
            "ground_truth": ground_truth,
            "context_snippet": context[:100],
        }

        print(f"[GPU {gpu_id} | 샘플 {sample_index}] 1. Baseline (Query-Only) 추론 중...")
        result_row["answer_baseline"] = generate_answer(base_model, tokenizer, question)
        print(f"[GPU {gpu_id} | 샘플 {sample_index}] 2. Baseline (Query + Context) 추론 중...")
        prompt_with_context = f"Based on the following context, please answer the question.\n\n[Context]\n{context}\n\n[Question]\n{question}"
        result_row["answer_baseline_with_context"] = generate_answer(base_model, tokenizer, prompt_with_context)

        for use_train_prompt in TRAINING_PROMPT_SETTINGS:
            for loss_masking_strategy in LOSS_MASKING_SETTINGS:
                
                if not use_train_prompt and loss_masking_strategy == "context_only":
                    continue

                if not use_train_prompt:
                    combined_key_suffix = "naive"
                    combined_display_name = "Naive (Loss on Context)"
                else:
                    combined_key_suffix = f"prompted_loss_{loss_masking_strategy}"
                    mask_display = "All" if loss_masking_strategy == "all" else "Context-Only"
                    combined_display_name = f"Prompted (Loss on {mask_display})"

                print(f"[GPU {gpu_id} | 샘플 {sample_index}] 3. PEFT 모드 시작: [{combined_display_name}]")
                train_data_dict = {}
                if use_train_prompt:
                    training_content = f"{TRAIN_PROMPT_PREFIX}\n{context}"
                    tokenized_data = tokenizer(training_content, truncation=True, max_length=MAX_CONTEXT_LENGTH)
                    labels = tokenized_data['input_ids'].copy()
                    
                    if loss_masking_strategy == "context_only":
                        prompt_only_tokens = tokenizer(f"{TRAIN_PROMPT_PREFIX}\n", add_special_tokens=True)
                        prompt_len_with_bos = len(prompt_only_tokens['input_ids'])
                        for k in range(prompt_len_with_bos):
                            if k < len(labels):
                                labels[k] = -100
                    
                    train_data_dict = {
                        "input_ids": tokenized_data['input_ids'],
                        "attention_mask": tokenized_data['attention_mask'],
                        "labels": labels
                    }
                else: 
                    tokenized_data = tokenizer(context, truncation=True, max_length=MAX_CONTEXT_LENGTH)
                    tokenized_data['labels'] = tokenized_data['input_ids'].copy()
                    train_data_dict = tokenized_data
                
                train_dataset = Dataset.from_dict({k: [v] for k, v in train_data_dict.items()})

                for adapter_type in ADAPTER_TYPES:
                    for num_epochs in EPOCH_SETTINGS:
                        
                        print(f"[GPU {gpu_id} | 샘플 {sample_index}]   -> 학습 시작: {adapter_type} / {num_epochs} E / {combined_display_name}")

                        peft_config = LoraConfig(
                            r=LORA_R,
                            lora_alpha=LORA_ALPHA,
                            target_modules=TARGET_MODULES,
                            lora_dropout=LORA_DROPOUT,
                            bias="none",
                            task_type="CAUSAL_LM",
                            use_dora=(adapter_type == "DoRA")
                        )
                        peft_model = get_peft_model(base_model, peft_config)
                        peft_model.train()
                        
                        output_dir = f"./results/temp_sample_{sample_index}/{adapter_type.lower()}_e{num_epochs}_{combined_key_suffix}"

                        training_args = TrainingArguments(
                            output_dir=output_dir,
                            per_device_train_batch_size=1,
                            num_train_epochs=num_epochs,
                            learning_rate=LEARNING_RATE,
                            logging_steps=1,
                            save_strategy="no",
                            report_to="none",
                            use_cpu=False
                        )
                        
                        trainer = Trainer(
                            model=peft_model,
                            args=training_args,
                            train_dataset=train_dataset,
                            data_collator=data_collator,
                        )
                        
                        trainer.train()

                        log_history = trainer.state.log_history
                        losses = [log['loss'] for log in log_history if 'loss' in log]
                        
                        peft_model.eval()
                        answer_peft = generate_answer(peft_model, tokenizer, question)
                        
                        key_base = f"{adapter_type.lower()}_e{num_epochs}_{combined_key_suffix}"
                        result_row[f"answer_{key_base}"] = answer_peft
                        result_row[f"max_loss_{key_base}"] = np.max(losses) if losses else None
                        result_row[f"min_loss_{key_base}"] = np.min(losses) if losses else None
                        result_row[f"avg_loss_{key_base}"] = np.mean(losses) if losses else None


                        peft_model.unload()
                        del peft_model
                        del trainer
                        torch.cuda.empty_cache()
                        
                        print(f"[GPU {gpu_id} | 샘플 {sample_index}]   -> 완료: {adapter_type} / {num_epochs} E / {combined_display_name}")

        print(f"[GPU {gpu_id} | 샘플 {sample_index}] ✨ 작업 완료.")
        
        del base_model
        del tokenizer
        del dataset
        torch.cuda.empty_cache()

        return result_row

    except Exception as e:
        print(f"[GPU {gpu_id} | 샘플 {sample_index}] 🚨 에러 발생: {e}")
        torch.cuda.empty_cache()
        # 실패 시에도 기본 정보는 반환
        return {
            "sample_index": sample_index,
            "gpu_id": gpu_id,
            "error": str(e)
        }


def main():
    print("="*80)
    print("🚀 다중 GPU 병렬 실험 스크립트 시작")
    print(f"   - 모델: {MODEL_ID}")
    print(f"   - 샘플 수: {NUM_SAMPLES}")
    print(f"   - 사용 GPU: {AVAILABLE_GPUS}")
    print(f"   - 동시 작업 수 (워커): {len(AVAILABLE_GPUS)}")
    print("="*80)

    tasks = []
    gpu_cycler = cycle(AVAILABLE_GPUS)
    
    for i in range(NUM_SAMPLES):
        sample_index = i * SAMPLE_STRIDE
        gpu_id = next(gpu_cycler)
        tasks.append((sample_index, gpu_id, DATASET_PATH))

    print("\n[메인] 다음 작업들을 풀(Pool)에 할당합니다:")
    for task in tasks:
        print(f"  - 샘플 인덱스: {task[0]}, 할당 GPU: {task[1]}")

    multiprocessing.set_start_method("spawn", force=True)
    
    all_results = []
    
    try:
        with multiprocessing.Pool(processes=len(AVAILABLE_GPUS)) as pool:
            print("\n[메인] 워커 프로세스에 작업 할당 시작... (완료까지 대기)")
            all_results = pool.map(process_sample_task, tasks)
            
        print("\n[메인] 모든 워커 프로세스 작업 완료.")

    except KeyboardInterrupt:
        print("\n[메인] 🚨 사용자에 의해 작업 중단. 풀을 종료합니다.")
        pool.terminate()
        pool.join()
        return
    except Exception as e:
        print(f"\n[메인] 🚨 병렬 처리 중 심각한 에러 발생: {e}")
        return

    if not all_results:
        print("[메인] 결과가 수집되지 않았습니다. 종료합니다.")
        return

    print(f"\n[메인] 총 {len(all_results)}개의 결과 수집 완료. CSV 파일 저장 중...")
    all_results.sort(key=lambda x: x.get("sample_index", -1))
    
    df = pd.DataFrame(all_results)
    pd.set_option('display.max_colwidth', None)
    
    try:
        df.to_csv(OUTPUT_CSV_FILENAME, index=False, encoding="utf-8-sig")
        print(f"\n[메인] ✨ 최종 결과가 '{OUTPUT_CSV_FILENAME}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"[메인] 🚨 CSV 파일 저장 실패: {e}")

    print("\n" + "="*80)
    print("📊 실험 결과 요약 (일부)")
    print("="*80)
    
    for result in all_results[:3]: 
        print("\n" + "-"*70)
        print(f"Sample Index: {result.get('sample_index', 'N/A')}")
        print(f"GPU ID: {result.get('gpu_id', 'N/A')}")
        print(f"Question: {result.get('question', 'N/A')}")
        print(f"Ground Truth: {result.get('ground_truth', 'N/A')}")
        print(f"Baseline (Query+Ctx): {result.get('answer_baseline_with_context', 'N/A')}")
        lora_key = "answer_lora_e10_prompted_loss_context_only"
        if lora_key in result:
             print(f"PEFT (Prompted): {result.get(lora_key, 'N/A')}")
        else:
            lora_key_naive = "answer_lora_e10_naive"
            if lora_key_naive in result:
                print(f"PEFT (Naive): {result.get(lora_key_naive, 'N/A')}")
        
        if "error" in result:
            print(f"🚨 ERROR: {result['error']}")

    print("\n[메인] 스크립트 실행 완료.")


if __name__ == "__main__":
    main()