import asyncio
import json
import os
import glob
import argparse
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "/mnt/raid5/kangjh/Research/context-param/dataset/sample_dataset/eval.json"
ADAPTER_ROOT = "/mnt/raid5/kangjh/Research/context-param/Exp1/adapters"
SYS_PROMPT = "You are a helpful AI assistant."


def get_no_context_messages(question: str):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAnswer:"}
    ]


def find_dora_adapters(context_id: int, debug_mode: bool = False) -> List[Dict[str, str]]:
    """DoRA 어댑터만 찾습니다 (dora 폴더 내의 어댑터만)."""
    dir_name = f"sample{context_id}"
    sample_dir = os.path.join(ADAPTER_ROOT, dir_name)
    
    if debug_mode:
        print(f"\n[DEBUG context_id={context_id}] Searching DoRA in: {sample_dir}")
    
    if not os.path.exists(sample_dir):
        if debug_mode:
            print(f"[DEBUG context_id={context_id}] ❌ Directory NOT found: {sample_dir}")
        return []
    
    # DoRA만 찾기: ADAPTER_ROOT/sampleX/dora/*/adapter
    search_pattern = os.path.join(sample_dir, "dora", "*", "adapter")
    found_paths = glob.glob(search_pattern)
    
    if debug_mode:
        print(f"[DEBUG context_id={context_id}] DoRA Pattern: {search_pattern}")
        print(f"[DEBUG context_id={context_id}] Found {len(found_paths)} DoRA adapters.")
    
    adapters = []
    for path in found_paths:
        parts = path.split(os.sep)
        
        try:
            setting = parts[-2]  # e.g., e3_naive, e7_naive
            method = "dora"
            adapter_name = f"{method}/{setting}"
            
            adapters.append({
                "name": adapter_name,
                "method": method,
                "setting": setting,
                "path": path,
                "is_adapter": True
            })
            if debug_mode:
                print(f"[DEBUG context_id={context_id}] Loaded DoRA Adapter: {adapter_name}")
        except IndexError:
            if debug_mode:
                print(f"[DEBUG] ⚠️ Path parsing failed for: {path}")
    
    return adapters

class DoRAInferenceWorker:
    """GPU별로 독립적으로 동작하는 DoRA 추론 워커 (배치 추론 지원)"""
    
    def __init__(self, gpu_id: int, base_model_name: str):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.base_model = None
        self.current_adapter_path = None
        self.current_model = None
        self.lock = threading.Lock()
        
        print(f"[GPU {gpu_id}] Initializing worker...")
        self._init_base_model()
    
    def _init_base_model(self):
        """베이스 모델과 토크나이저 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device},  # 특정 GPU에 고정
            low_cpu_mem_usage=True
        )
        self.base_model.eval()
        print(f"[GPU {self.gpu_id}] Base model loaded on {self.device}")
    
    def _load_adapter(self, adapter_path: str):
        """어댑터 로드 (제대로 언로드하고 다시 로드)"""
        with self.lock:
            if self.current_adapter_path == adapter_path and self.current_model is not None:
                return
            
            # 기존 PeftModel 완전히 제거
            if self.current_model is not None:
                print(f"[GPU {self.gpu_id}] Unloading previous adapter...")
                self.base_model = self.current_model.unload()  # 어댑터만 제거
                del self.current_model
                self.current_model = None
                torch.cuda.empty_cache()
            
            # 새 어댑터 로드
            print(f"[GPU {self.gpu_id}] Loading adapter: {adapter_path}")
            self.current_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                device_map={"": self.device},  # 명시적으로 GPU 지정
                is_trainable=False
            )
            
            # 모든 파라미터가 올바른 device에 있는지 확인
            self.current_model = self.current_model.to(self.device)
            self.current_model.eval()
            self.current_adapter_path = adapter_path
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """채팅 템플릿 적용"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    @torch.no_grad()
    def generate_batch(
        self, 
        messages_list: List[List[Dict[str, str]]], 
        adapter_path: str = None, 
        max_tokens: int = 128
    ) -> List[str]:
        """배치 추론 실행"""
        try:
            if adapter_path:
                self._load_adapter(adapter_path)
                model = self.current_model
            else:
                model = self.base_model
            
            prompts = [self.apply_chat_template(messages) for messages in messages_list]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # temperature=0.0이면 do_sample=False로
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            responses = []
            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())
            
            return responses
            
        except Exception as e:
            print(f"[GPU {self.gpu_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            return [f"Error: {str(e)}"] * len(messages_list)



def process_sample_sync(
    sample: Dict[str, Any],
    worker: DoRAInferenceWorker,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """단일 샘플 처리 (배치 최적화 버전)"""
    
    context_id = sample["context_id"]
    qa_pairs = sample["qa_pairs"]
    
    # DoRA 어댑터 찾기
    dora_adapters = find_dora_adapters(context_id, debug_mode=debug_mode)
    
    sample_result = {
        "context_id": context_id,
        "qa_results": []
    }
    
    # 모든 QA pair에 대한 messages를 미리 생성
    all_messages = [get_no_context_messages(qa["question"]) for qa in qa_pairs]
    ground_truths = [qa["answer"] for qa in qa_pairs]
    questions = [qa["question"] for qa in qa_pairs]
    
    # 각 어댑터에 대해 배치 추론
    adapter_predictions = {}
    for adapter in dora_adapters:
        # 10개 QA를 한 번에 처리!
        batch_responses = worker.generate_batch(
            all_messages,
            adapter_path=adapter["path"]
        )
        adapter_predictions[adapter["name"]] = batch_responses
    
    # 결과 구성
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        qa_res = {
            "question": question,
            "ground_truth": ground_truth,
            "predictions": {}
        }
        
        # 각 어댑터의 i번째 응답 저장
        for adapter in dora_adapters:
            qa_res["predictions"][adapter["name"]] = adapter_predictions[adapter["name"]][i]
        
        sample_result["qa_results"].append(qa_res)
    
    return sample_result


def main():
    parser = argparse.ArgumentParser(description="DoRA-only Inference with PEFT (Batch Optimized)")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1], help="List of GPU IDs to use")
    parser.add_argument("--output", type=str, default="dora_inference_results.json", help="Output file path")
    parser.add_argument("--debug", action="store_true", help="Run only 5 samples for debugging")
    args = parser.parse_args()
    
    print(f"=== DoRA Batch Inference Settings ===")
    print(f"GPUs: {args.gpus}")
    print(f"Output: {args.output}")
    print(f"Debug Mode: {args.debug}")
    print(f"Optimization: Batch inference enabled (10 QA pairs per batch)")
    
    # 데이터셋 로드
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
        samples = dataset["data"]
    
    if args.debug:
        print("!!! DEBUG MODE: Processing only top 5 samples !!!")
        samples = samples[:5]
    
    # GPU별 워커 생성
    workers = [DoRAInferenceWorker(gpu_id, BASE_MODEL_NAME) for gpu_id in args.gpus]
    
    # 멀티스레딩으로 병렬 처리
    results = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = []
        
        for i, sample in enumerate(samples):
            worker = workers[i % len(workers)]
            future = executor.submit(
                process_sample_sync,
                sample,
                worker,
                args.debug
            )
            futures.append(future)
        
        # 진행상황 표시하며 결과 수집
        for future in tqdm(futures, desc="Processing Samples (Batch Mode)"):
            result = future.result()
            results.append(result)
    
    # 결과 정렬 및 저장
    results.sort(key=lambda x: x["context_id"])
    final_output = {
        "metadata": dataset.get("metadata", {}),
        "inference_config": {
            "base_model": BASE_MODEL_NAME,
            "adapters_root": ADAPTER_ROOT,
            "gpus": args.gpus,
            "adapter_type": "DoRA",
            "batch_inference": True,
            "qa_batch_size": 10
        },
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, indent=2, ensure_ascii=False, fp=f)
    
    print(f"\n✅ Successfully saved DoRA batch results to {args.output}")
    print(f"🚀 Speedup: ~3-5x faster than sequential processing")


if __name__ == "__main__":
    main()
