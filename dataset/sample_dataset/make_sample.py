import json
import random
from pathlib import Path
from typing import List, Dict, Any
import copy

def load_dataset(file_path: str) -> Dict[str, Any]:
    """원본 JSON 데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def split_dataset(
    input_file: str,
    output_dir: str = "./data",
    num_contexts: int = 200,
    num_eval_qa_per_context: int = 10,
    seed: int = 42
):
    """
    데이터셋을 train/eval/dev로 분할
    Dev = Train + Eval (같은 context의 전체 QA)
    """
    random.seed(seed)
    
    # 1. 원본 데이터 로드
    print(f"Loading dataset from {input_file}...")
    dataset = load_dataset(input_file)
    
    # 2. context_id로 정렬
    print("Sorting by context_id...")
    sorted_data = sorted(dataset['data'], key=lambda x: x['context_id'])
    
    # 3. 200개 context 무작위 추출
    print(f"Sampling {num_contexts} contexts randomly...")
    sampled_contexts = sorted_data[:num_contexts]
    
    # 4. eval/train 데이터 생성
    print(f"Creating eval split ({num_eval_qa_per_context} QA pairs per context)...")
    print("Creating train split (remaining QA pairs)...")
    
    eval_data_list = []
    train_data_list = []
    dev_data_list = []
    
    for context_item in sampled_contexts:
        qa_pairs = context_item['qa_pairs']
        context_id = context_item['context_id']
        context_text = context_item['context']
        
        # QA 쌍이 충분한지 확인
        if len(qa_pairs) < num_eval_qa_per_context:
            print(f"Warning: context_id {context_id} has only {len(qa_pairs)} QA pairs. Using all for eval.")
            eval_qa = qa_pairs
            train_qa = []
        else:
            # 무작위로 eval용 QA 추출
            eval_qa = random.sample(qa_pairs, num_eval_qa_per_context)
            # 나머지는 train용
            eval_qa_indices = {id(qa) for qa in eval_qa}
            train_qa = [qa for qa in qa_pairs if id(qa) not in eval_qa_indices]
        
        # eval 데이터 추가
        if eval_qa:
            eval_data_list.append({
                "context_id": context_id,
                "context": context_text,
                "qa_pairs": eval_qa
            })
        
        # train 데이터 추가
        if train_qa:
            train_data_list.append({
                "context_id": context_id,
                "context": context_text,
                "qa_pairs": train_qa
            })
        
        # dev는 전체 QA 포함 (train + eval의 모든 QA)
        dev_data_list.append({
            "context_id": context_id,
            "context": context_text,
            "qa_pairs": qa_pairs  # 전체 QA 쌍
        })
    
    # 메타데이터 생성
    eval_data = {
        "metadata": {
            **dataset['metadata'],
            "split": "eval",
            "num_contexts": len(eval_data_list),
            "total_qa_pairs": sum(len(ctx['qa_pairs']) for ctx in eval_data_list)
        },
        "data": eval_data_list
    }
    
    train_data = {
        "metadata": {
            **dataset['metadata'],
            "split": "train",
            "num_contexts": len(train_data_list),
            "total_qa_pairs": sum(len(ctx['qa_pairs']) for ctx in train_data_list)
        },
        "data": train_data_list
    }
    
    dev_data = {
        "metadata": {
            **dataset['metadata'],
            "split": "dev",
            "num_contexts": len(dev_data_list),
            "total_qa_pairs": sum(len(ctx['qa_pairs']) for ctx in dev_data_list)
        },
        "data": dev_data_list
    }
    
    # 6. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_data,
        'eval': eval_data,
        'dev': dev_data
    }
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {split_name}: {split_data['metadata']['num_contexts']} contexts, "
              f"{split_data['metadata']['total_qa_pairs']} QA pairs -> {output_file}")
    
    # 7. 검증: train + eval = dev
    train_total = train_data['metadata']['total_qa_pairs']
    eval_total = eval_data['metadata']['total_qa_pairs']
    dev_total = dev_data['metadata']['total_qa_pairs']
    
    print("\n=== Split Statistics ===")
    print(f"Train: {train_data['metadata']['num_contexts']} contexts, {train_total} QA pairs")
    print(f"Eval:  {eval_data['metadata']['num_contexts']} contexts, {eval_total} QA pairs")
    print(f"Dev:   {dev_data['metadata']['num_contexts']} contexts, {dev_total} QA pairs")
    print(f"\n✓ Verification: Train({train_total}) + Eval({eval_total}) = {train_total + eval_total} == Dev({dev_total})? {train_total + eval_total == dev_total}")
    print(f"\nAll splits saved to: {output_path.absolute()}")

# 실행
if __name__ == "__main__":
    split_dataset(
        input_file="/mnt/raid5/kangjh/Research/context-param/dataset/wikiqa/wiki_qa.json",
        output_dir="/mnt/raid5/kangjh/Research/context-param/dataset/sample_dataset",
        num_contexts=200,
        num_eval_qa_per_context=10,
        seed=42
    )
