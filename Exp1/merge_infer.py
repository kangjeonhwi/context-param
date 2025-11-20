import json
import os

def merge_inference_results(base_path, dora_path, output_path):
    print(f"Loading base results from: {base_path}")
    with open(base_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    print(f"Loading DoRA results from: {dora_path}")
    with open(dora_path, 'r', encoding='utf-8') as f:
        dora_data = json.load(f)
    print("Building lookup map for DoRA results...")
    dora_lookup = {}
    
    for item in dora_data['results']:
        ctx_id = item['context_id']
        
        if ctx_id not in dora_lookup:
            dora_lookup[ctx_id] = {}
            
        for qa in item['qa_results']:
            question_text = qa['question']
            # Key: question, Value: predictions dictionary
            dora_lookup[ctx_id][question_text] = qa['predictions']

    print("Merging predictions...")
    merged_count = 0
    for item in base_data['results']:
        ctx_id = item['context_id']
        
        if ctx_id in dora_lookup:
            for qa in item['qa_results']:
                question_text = qa['question']

                if question_text in dora_lookup[ctx_id]:
                    dora_preds = dora_lookup[ctx_id][question_text]

                    qa['predictions'].update(dora_preds)
                    merged_count += 1
                else:
                    pass
    print(f"Total QA pairs merged: {merged_count}")
    print(f"Saving merged results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, indent=2, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    # 파일 경로 설정
    BASE_FILE = "/mnt/raid5/kangjh/Research/context-param/Exp1/outputs/lora_inference_results.json"
    DORA_FILE = "/mnt/raid5/kangjh/Research/context-param/Exp1/outputs/dora_inference_results.json"
    OUTPUT_FILE = "/mnt/raid5/kangjh/Research/context-param/Exp1/outputs/merged_inference_results.json"

    # 파일 존재 여부 확인
    if os.path.exists(BASE_FILE) and os.path.exists(DORA_FILE):
        merge_inference_results(BASE_FILE, DORA_FILE, OUTPUT_FILE)
    else:
        print("Error: One or more input files not found.")
        if not os.path.exists(BASE_FILE): print(f"Missing: {BASE_FILE}")
        if not os.path.exists(DORA_FILE): print(f"Missing: {DORA_FILE}")