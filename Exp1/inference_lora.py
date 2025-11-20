import asyncio
import json
import os
import glob
import argparse
import aiohttp
from typing import List, Dict, Any
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "/mnt/raid5/kangjh/Research/context-param/dataset/sample_dataset/eval.json"
ADAPTER_ROOT = "/mnt/raid5/kangjh/Research/context-param/Exp1/adapters"
SYS_PROMPT = "You are a helpful AI assistant."

def get_no_context_messages(question: str):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAnswer:"}
    ]

def get_with_context_messages(context: str, question: str):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ]

def find_adapters(context_id: int, debug_mode: bool = False) -> List[Dict[str, str]]:
    """
    Debug 모드가 켜져 있으면 경로 탐색 과정을 상세히 출력합니다.
    """
    # 1. 탐색할 디렉토리 설정 (폴더명 규칙 확인용)
    # 사용자의 디렉토리 구조가 'sample0', 'sample1' ... 이라고 가정
    dir_name = f"sample{context_id}" 
    sample_dir = os.path.join(ADAPTER_ROOT, dir_name)
    
    if debug_mode:
        print(f"\n[DEBUG context_id={context_id}] Searching in: {sample_dir}")

    # 2. 디렉토리 존재 여부 확인
    if not os.path.exists(sample_dir):
        if debug_mode:
            print(f"[DEBUG context_id={context_id}] ❌ Directory NOT found: {sample_dir}")
            # 혹시 sample_0 처럼 언더바가 들어가는지, 혹은 경로가 틀렸는지 확인하기 위해 상위 디렉토리 리스트 출력
            if os.path.exists(ADAPTER_ROOT):
                print(f"[DEBUG] Available dirs in root: {os.listdir(ADAPTER_ROOT)[:10]} ...")
        return []

    # 3. Glob 패턴 매칭
    # 패턴: ADAPTER_ROOT/sampleX / * (method) / * (setting) / adapter
    search_pattern = os.path.join(sample_dir, "*", "*", "adapter")
    found_paths = glob.glob(search_pattern)

    if debug_mode:
        print(f"[DEBUG context_id={context_id}] Pattern used: {search_pattern}")
        print(f"[DEBUG context_id={context_id}] Found {len(found_paths)} adapter paths.")
        
        if len(found_paths) == 0:
            try:
                subdirs = os.listdir(sample_dir)
                print(f"[DEBUG context_id={context_id}] Subdirectories inside {dir_name}: {subdirs}")
                if subdirs:
                    # 한 단계 더 들어가서 확인
                    first_sub = os.path.join(sample_dir, subdirs[0])
                    if os.path.isdir(first_sub):
                        print(f"[DEBUG context_id={context_id}] Inside {subdirs[0]}: {os.listdir(first_sub)}")
            except Exception as e:
                print(f"[DEBUG] Error listing dirs: {e}")

    adapters = []
    for path in found_paths:
        parts = path.split(os.sep)
        
        try:
            setting = parts[-2]
            method = parts[-3]
            adapter_name = f"{method}/{setting}"
            if method == 'dora' :
                continue
            adapters.append({
                "name": adapter_name,
                "method": method,
                "setting": setting,
                "path": path,
                "is_adapter": True
            })
            if debug_mode:
                print(f"[DEBUG context_id={context_id}] Loaded Adapter: {adapter_name}")
        except IndexError:
            if debug_mode:
                print(f"[DEBUG] ⚠️ Path parsing failed for: {path}")

    return adapters

async def get_model_response(
    client: AsyncOpenAI, 
    model_path: str, 
    messages: List[Dict[str, str]],
    max_tokens: int = 128
) -> str:
    """
    OpenAI API 호환 호출. 
    vLLM에서 LoRA를 사용할 때는 model 인자에 로컬 절대 경로를 넣으면 로드됩니다.
    """
    try:
        response = await client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0, # Deterministic evaluation
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 어댑터 로드 실패 등의 에러 처리
        return f"Error: {str(e)}"
    
async def load_adapter_to_server(port: int, adapter_name: str, adapter_path: str):
    """런타임에 LoRA 어댑터를 서버에 로드"""
    url = f"http://localhost:{port}/v1/load_lora_adapter"
    payload = {
        "lora_name": adapter_name,
        "lora_path": adapter_path
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Failed to load {adapter_name}: {await response.text()}")
                return False
    return True


async def process_sample(
    sample: Dict[str, Any], 
    port: int, 
    sem: asyncio.Semaphore,
    loaded_adapters: set,  # 이미 로드된 어댑터 추적
    debug_mode: bool = False
) -> Dict[str, Any]:
    
    context_id = sample["context_id"]
    context_text = sample["context"]
    qa_pairs = sample["qa_pairs"]
    
    client = AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY"
    )

    adapters = find_adapters(context_id, debug_mode=debug_mode)
    
    # 어댑터 로드 (한 번만)
    for adapter in adapters:
        adapter_key = f"{port}:{adapter['name']}"
        if adapter_key not in loaded_adapters:
            success = await load_adapter_to_server(port, adapter['name'], adapter['path'])
            if success:
                loaded_adapters.add(adapter_key)
    
    sample_result = {
        "context_id": context_id,
        "qa_results": []
    }

    async with sem:
        for qa in qa_pairs:
            question = qa["question"]
            ground_truth = qa["answer"]
            
            qa_res = {
                "question": question,
                "ground_truth": ground_truth,
                "predictions": {}
            }
            
            # Base model
            messages_no_ctx = get_no_context_messages(question)
            qa_res["predictions"]["base_no_context"] = await get_model_response(
                client, BASE_MODEL_NAME, messages_no_ctx
            )
            
            messages_ctx = get_with_context_messages(context_text, question)
            qa_res["predictions"]["base_with_context"] = await get_model_response(
                client, BASE_MODEL_NAME, messages_ctx
            )
            
            # LoRA adapters - model 파라미터에 어댑터 이름 사용
            for adapter in adapters:
                qa_res["predictions"][adapter["name"]] = await get_model_response(
                    client, adapter["name"], messages_no_ctx  # 경로 대신 이름 사용
                )
            
            sample_result["qa_results"].append(qa_res)
            
    await client.close()
    return sample_result

async def main():
    parser = argparse.ArgumentParser(description="Distributed Inference for LoRA Evaluation")
    parser.add_argument("--ports", nargs="+", type=int, default=[8000, 8001, 8002, 8003], help="List of vLLM serving ports")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output file path")
    parser.add_argument("--debug", action="store_true", help="Run only 2 samples for debugging")
    args = parser.parse_args()

    print(f"=== Inference Settings ===")
    print(f"Ports: {args.ports}")
    print(f"Output: {args.output}")
    print(f"Debug Mode: {args.debug}")
    
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
        samples = dataset["data"]

    if args.debug:
        print("!!! DEBUG MODE: Processing only top 5 samples !!!")
        samples = samples[:5]

    # 각 포트별로 로드된 어댑터 추적
    loaded_adapters_per_port = {port: set() for port in args.ports}
    
    tasks = []
    semaphore = asyncio.Semaphore(16 * len(args.ports))

    for i, sample in enumerate(samples):
        assigned_port = args.ports[i % len(args.ports)]
        task = asyncio.create_task(
            process_sample(
                sample, 
                assigned_port, 
                semaphore, 
                loaded_adapters_per_port[assigned_port],  # set 전달
                args.debug  # debug_mode는 별도로
            )
        )
        tasks.append(task)

    results = []
    for f in tqdm.as_completed(tasks, desc="Processing Samples"):
        res = await f
        results.append(res)

    results.sort(key=lambda x: x["context_id"])
    final_output = {
        "metadata": dataset.get("metadata", {}),
        "inference_config": {
            "base_model": BASE_MODEL_NAME,
            "adapters_root": ADAPTER_ROOT,
            "ports": args.ports
        },
        "results": results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully saved results to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())