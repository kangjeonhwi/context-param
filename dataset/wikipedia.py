from datasets import load_dataset
import json
import os
import unicodedata

from dir_path import DATA_DIR


WIKI_DATASET_NAME = "HuggingFaceFW/finewiki"
WIKI_DATASET_SUBSET = "en"
MIN_ARTICLE_LENGTH = 3000
MAX_ARTICLE_LENGTH = 5000
NUM_SAMPLES = 1000

def clean_unicode(text):
    text = unicodedata.normalize('NFKC', text)
    return text

print("Loading and filtering dataset...")
wiki_dataset = load_dataset(WIKI_DATASET_NAME, WIKI_DATASET_SUBSET, split="train", streaming=True)

filtered_dataset = wiki_dataset.filter(
    lambda x: (len(x['text']) > MIN_ARTICLE_LENGTH and len(x['text']) < MAX_ARTICLE_LENGTH)
)

contexts = []
for i, example in enumerate(filtered_dataset):
    if i >= NUM_SAMPLES:
        break
    cleaned_text = clean_unicode(example['text'])
    contexts.append(cleaned_text)
    if (i + 1) % 100 == 0:
        print(f"Sampled {i+1}/{NUM_SAMPLES} articles...")

print(contexts[0])
print(f"Successfully sampled {len(contexts)} long articles.")

datasets = []
for i, context in enumerate(contexts):
    datasets.append({
        "id": i,
        "context": context
    })

save_dir = os.path.join(DATA_DIR, "wikidata")
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "wiki_contexts.json"), 'w', encoding='utf-8') as f:
    json.dump(datasets, f, ensure_ascii=False, indent=4)

print(len(datasets), "articles saved to wiki_contexts.json")