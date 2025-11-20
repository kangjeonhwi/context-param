import os
import json
import torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
import gc



# ======================== 설정 ========================
@dataclass
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path: str = "/mnt/raid5/kangjh/Research/context-param/dataset/sample_dataset/eval.json"
    output_base_dir: str = "/mnt/raid5/kangjh/Research/context-param/Exp1/adapters"
    metadata_file: str = "./adapter_metadata.json"
    
    num_samples: int = 200
    sample_stride: int = 1
    
    # LoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # 학습 설정
    learning_rate: float = 2e-4
    max_context_length: int = 4096
    per_device_train_batch_size: int = 1
    
    # 실험 조합
    epoch_settings: List[int] = None
    adapter_types: List[str] = None
    training_prompt_settings: List[bool] = None
    loss_masking_settings: List[str] = None
    
    train_prompt_prefix: str = "Please memorize the following context carefully and answer the question based on it. Context: "
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["gate_proj", "up_proj", "down_proj"]
        if self.epoch_settings is None:
            self.epoch_settings = [3, 7, 15]
        if self.adapter_types is None:
            self.adapter_types = ["LoRA", "DoRA"]
        if self.training_prompt_settings is None:
            self.training_prompt_settings = [False, True]
        if self.loss_masking_settings is None:
            self.loss_masking_settings = ["all", "context_only"]



# ======================== 유틸리티 함수 ========================
def prepare_training_data(
    sample: Dict,
    tokenizer,
    config: ExperimentConfig,
    use_train_prompt: bool,
    loss_masking_strategy: str
) -> Dict:
    """
    단일 샘플에 대한 학습 데이터 준비
    """
    context = sample['context']
    
    if use_train_prompt:
        training_content = f"{config.train_prompt_prefix}\n{context}"
        tokenized_data = tokenizer(
            training_content, 
            truncation=True, 
            max_length=config.max_context_length
        )
        labels = tokenized_data['input_ids'].copy()
        
        # Context-only loss masking
        if loss_masking_strategy == "context_only":
            prompt_only_tokens = tokenizer(
                f"{config.train_prompt_prefix}\n", 
                add_special_tokens=True
            )
            prompt_len_with_bos = len(prompt_only_tokens['input_ids'])
            for k in range(prompt_len_with_bos):
                if k < len(labels):
                    labels[k] = -100
        
        train_data_dict = {
            "input_ids": tokenized_data['input_ids'],
            "attention_mask": tokenized_data['attention_mask'],
            "labels": labels
        }
    else:  # Naive: loss on context only
        tokenized_data = tokenizer(
            context, 
            truncation=True, 
            max_length=config.max_context_length
        )
        train_data_dict = {
            "input_ids": tokenized_data['input_ids'],
            "attention_mask": tokenized_data['attention_mask'],
            "labels": tokenized_data['input_ids'].copy()
        }
    
    return train_data_dict



def safe_delete_adapter(model: PeftModel, adapter_name: str = None):
    """
    LoRA 어댑터를 안전하게 제거하고 메모리 해제
    """
    try:
        if adapter_name:
            # 특정 어댑터 삭제
            if hasattr(model, 'delete_adapter'):
                model.delete_adapter(adapter_name)
        else:
            # 모든 어댑터 unload
            if hasattr(model, 'unload'):
                model.unload()
        
        # 명시적 가비지 컬렉션
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️ 어댑터 삭제 중 경고: {e}")



def get_experiment_path(
    base_dir: str,
    sample_idx: int,
    adapter_type: str,
    num_epochs: int,
    use_train_prompt: bool,
    loss_masking_strategy: str
) -> tuple[str, str]:
    """
    깔끔한 디렉토리 구조 생성: sample{idx}/{adapter_type}/e{epochs}_{suffix}/
    
    Returns:
        (output_dir, experiment_key)
    """
    # Suffix 생성
    if not use_train_prompt:
        suffix = "naive"
    else:
        suffix = f"prompted_loss_{loss_masking_strategy}"
    
    # 계층적 디렉토리 구조
    adapter_dir = adapter_type.lower()
    epoch_config_dir = f"e{num_epochs}_{suffix}"
    
    output_dir = os.path.join(
        base_dir,
        f"sample{sample_idx}",
        adapter_dir,
        epoch_config_dir
    )
    
    # 실험 키 (메타데이터용)
    experiment_key = f"sample{sample_idx}_{adapter_dir}_e{num_epochs}_{suffix}"
    
    return output_dir, experiment_key



# ======================== 메인 학습 함수 ========================
def train_single_adapter(
    base_model,
    tokenizer,
    train_dataset: Dataset,
    config: ExperimentConfig,
    adapter_type: str,
    num_epochs: int,
    sample_idx: int,
    use_train_prompt: bool,
    loss_masking_strategy: str,
    accelerator: Accelerator
) -> Dict:
    """
    단일 LoRA 어댑터 학습 및 저장
    """
    # 출력 경로 생성 (새로운 계층적 구조)
    output_dir, experiment_key = get_experiment_path(
        config.output_base_dir,
        sample_idx,
        adapter_type,
        num_epochs,
        use_train_prompt,
        loss_masking_strategy
    )
    
    print(f"\n{'='*70}")
    print(f"🔧 학습 시작: {experiment_key}")
    print(f"   📁 저장 경로: {output_dir}")
    print(f"{'='*70}")
    
    # LoRA Config 생성
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=(adapter_type == "DoRA")
    )
    
    # PEFT 모델 생성
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=0,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        bf16=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 학습 실행
    trainer.train()
    
    # Loss 정보 추출
    log_history = trainer.state.log_history
    losses = [log['loss'] for log in log_history if 'loss' in log]
    
    loss_stats = {
        "max_loss": float(max(losses)) if losses else None,
        "min_loss": float(min(losses)) if losses else None,
        "avg_loss": float(sum(losses) / len(losses)) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
    }
    
    # 어댑터 저장
    adapter_save_path = os.path.join(output_dir, "adapter")
    peft_model.save_pretrained(adapter_save_path)
    print(f"✅ 어댑터 저장 완료: {adapter_save_path}")
    
    # 메타데이터 생성
    metadata = {
        "experiment_key": experiment_key,
        "adapter_path": adapter_save_path,
        "output_dir": output_dir,
        "adapter_type": adapter_type,
        "num_epochs": num_epochs,
        "loss_stats": loss_stats,
    }
    
    # 개별 메타데이터 저장
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 어댑터 unload
    print(f"🧹 어댑터 unload 중...")
    safe_delete_adapter(peft_model)
    
    del peft_model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ 학습 완료 및 메모리 정리: {experiment_key}\n")
    
    return metadata



def main():
    print("\n" + "="*80)
    print("🚀 LoRA 어댑터 학습 스크립트 (Accelerate 활용)")
    print("="*80 + "\n")
    
    # Config 로드
    config = ExperimentConfig()
    
    # Accelerator 초기화
    accelerator = Accelerator()
    print(f"📊 Accelerator 정보:")
    print(f"  - 사용 가능 GPU: {torch.cuda.device_count()}")
    print(f"  - 현재 프로세스: {accelerator.process_index}/{accelerator.num_processes}")
    print(f"  - Device: {accelerator.device}\n")
    
    # Base Model & Tokenizer 로드
    print(f"🔄 Base Model 로드 중: {config.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map={"": accelerator.device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    base_model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Base Model 로드 완료\n")
    
    # 데이터셋 로드
    print(f"📂 데이터셋 로드 중: {config.dataset_path}")
    dataset = load_dataset(
        "json", 
        data_files=config.dataset_path, 
        field="data"
    )['train']
    print(f"✅ 데이터셋 로드 완료: {len(dataset)} 샘플\n")
    
    # 메타데이터 저장용 리스트
    all_metadata = []
    
    # 샘플별 학습 루프
    for sample_idx in range(config.num_samples):
        actual_idx = sample_idx * config.sample_stride
        sample = dataset[actual_idx]
        sample_id = sample['context_id']
        
        print(f"\n{'#'*80}")
        print(f"📝 샘플 {sample_idx}/{config.num_samples} 처리 중 (실제 인덱스: {actual_idx})")
        print(f"{'#'*80}")
        
        # 실험 조합별 학습
        for use_train_prompt in config.training_prompt_settings:
            for loss_masking_strategy in config.loss_masking_settings:
                
                # Naive 설정 시 context_only는 스킵
                if not use_train_prompt and loss_masking_strategy == "context_only":
                    continue
                
                # 학습 데이터 준비
                train_data_dict = prepare_training_data(
                    sample, tokenizer, config,
                    use_train_prompt, loss_masking_strategy
                )
                train_dataset = Dataset.from_dict({
                    k: [v] for k, v in train_data_dict.items()
                })
                
                # Adapter Type & Epochs 조합
                for adapter_type in config.adapter_types:
                    if adapter_type == "DoRA" and use_train_prompt :
                        continue

                    for num_epochs in config.epoch_settings:
                        
                        # Base model 상태 체크
                        if isinstance(base_model, PeftModel):
                            print(f"⚠️ 경고: Base model에 어댑터가 남아있음. 강제 정리 중...")
                            safe_delete_adapter(base_model)
                        
                        # 학습 실행
                        try:
                            metadata = train_single_adapter(
                                base_model, tokenizer, train_dataset,
                                config, adapter_type, num_epochs,
                                actual_idx, use_train_prompt, loss_masking_strategy,
                                accelerator
                            )
                            
                            # 추가 메타데이터
                            metadata['context_id'] = sample_id
                            metadata['use_train_prompt'] = use_train_prompt
                            metadata['loss_masking_strategy'] = loss_masking_strategy
                            
                            all_metadata.append(metadata)
                            
                        except Exception as e:
                            print(f"🚨 에러 발생: sample{actual_idx}_{adapter_type}_e{num_epochs}")
                            print(f"   {str(e)}")
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
    
    # 전체 메타데이터 저장
    print(f"\n{'='*80}")
    print(f"💾 전체 메타데이터 저장 중...")
    with open(config.metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 메타데이터 저장 완료: {config.metadata_file}")
    
    # CSV 요약 저장
    df = pd.DataFrame(all_metadata)
    summary_csv = config.metadata_file.replace('.json', '_summary.csv')
    df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 요약 CSV 저장 완료: {summary_csv}")
    
    print(f"\n{'='*80}")
    print(f"🎉 전체 학습 완료!")
    print(f"   - 총 어댑터 수: {len(all_metadata)}")
    print(f"   - 저장 위치: {config.output_base_dir}")
    print(f"{'='*80}\n")



if __name__ == "__main__":
    main()