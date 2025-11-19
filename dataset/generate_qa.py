import json
import unicodedata
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI

class ContextQAGenerator:
    """QA pair generation for context parameterization dataset."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_workers: int = 30,
        debug: bool = False
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_workers = max_workers
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] Initialized with model={model_name}, max_workers={max_workers}")

    @staticmethod
    def clean_unicode(text: str) -> str:
        """Normalize unicode characters in text."""
        return unicodedata.normalize('NFKC', text)

    def load_contexts(self, json_path: str) -> List[Dict]:
        """Load context list from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.debug:
            print(f"[DEBUG] Loaded {len(data)} contexts from {json_path}")
        return data

    def generate_qa_pairs(
        self,
        contexts: List[Dict],
        output_path: str = "setting1_qa_pairs.json",
        batch_size: int = 10
    ):
        """Generate QA pairs for all contexts and write results to file."""
        print("\n=== QA Pair Generation ===")
        print(f"Model: {self.model_name}, Server: {self.client.base_url}")
        print(f"Total contexts: {len(contexts)}")

        results = []
        total_qa_pairs = 0
        failed_contexts = []

        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(contexts)-1)//batch_size + 1

            print(f"\nProcessing batch {batch_num}/{total_batches}")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._generate_qa_for_context,
                        item['id'],
                        item['context']
                    ): item for item in batch
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_num}"):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            total_qa_pairs += result['num_pairs']
                            if self.debug and result['num_pairs'] > 0:
                                print(f"\n[DEBUG] Context {result['context_id']}: {result['num_pairs']} QA pairs generated")
                        else:
                            failed_contexts.append(futures[future]['id'])
                    except Exception as e:
                        item = futures[future]
                        print(f"\n[ERROR] Context {item['id']}: {e}")
                        failed_contexts.append(item['id'])

        metadata = {
            "total_contexts": len(contexts),
            "successful_contexts": len(results),
            "failed_contexts": len(failed_contexts),
            "total_qa_pairs": total_qa_pairs,
            "average_qa_per_context": total_qa_pairs / len(results) if results else 0,
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        output_data = {
            "metadata": metadata,
            "failed_context_ids": failed_contexts,
            "data": results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print("\n===== Generation Finished =====")
        print(f"Processed: {len(results)}/{len(contexts)}")
        print(f"QA pairs generated: {total_qa_pairs}")
        print(f"Average QA/context: {metadata['average_qa_per_context']:.2f}")
        print(f"Failed contexts: {len(failed_contexts)}")
        print(f"Output: {output_path}\n")

        return results

    def _generate_qa_for_context(
        self,
        context_id: int,
        context: str,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """Generate QA pairs for a single context, with retry and robust parsing."""
        for attempt in range(max_retries):
            try:
                if self.debug and attempt > 0:
                    print(f"[DEBUG] Retry {attempt + 1}/{max_retries} for context {context_id}")
                prompt = self._create_qa_generation_prompt(context)
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=4096,
                    stop=None
                )
                generated_text = response.choices[0].text.strip()
                if self.debug:
                    print(f"\n[DEBUG] Raw response (context {context_id}): {generated_text[:500]}...")
                qa_pairs = self._parse_qa_pairs_robust(generated_text, context_id)
                if qa_pairs:
                    return {
                        "context_id": context_id,
                        "context": context,
                        "qa_pairs": qa_pairs,
                        "num_pairs": len(qa_pairs),
                        "parsing_success": True
                    }
                else:
                    if attempt < max_retries - 1:
                        print(f"[WARNING] Context {context_id}: No QA pairs parsed. Retrying...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"[ERROR] Context {context_id}: Failed after {max_retries} attempts")
                        return {
                            "context_id": context_id,
                            "context": context,
                            "qa_pairs": [],
                            "num_pairs": 0,
                            "parsing_success": False,
                            "error": "No valid QA pairs could be extracted",
                            "raw_response": generated_text[:500] if self.debug else None
                        }
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[ERROR] Context {context_id}, attempt {attempt + 1}: {e}. Retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"[ERROR] Context {context_id}: Failed after {max_retries} attempts: {e}")
                    return {
                        "context_id": context_id,
                        "context": context,
                        "qa_pairs": [],
                        "num_pairs": 0,
                        "parsing_success": False,
                        "error": str(e)
                    }
        return None

    def _create_qa_generation_prompt(self, context: str) -> str:
        return f"""You are an expert AI assistant specialized in creating high-quality question-answer (QA) pairs from a given text.

Your task:
1. Carefully analyze the provided 'Context'.
2. Generate QA pairs that comprehensively cover ALL key information, facts, concepts, and details.
3. The number of QA pairs is FLEXIBLE:
   - Short, simple contexts may need only 1-3 pairs
   - Dense, information-rich contexts may need 10+ pairs
   - Use your judgment based on the information density
4. Each answer must be extracted STRICTLY from the provided Context (no external knowledge).
5. Ensure diversity in the QA pairs (different aspects, different types of questions).

Context:
{context}

CRITICAL: Your response must be a valid JSON array only. No markdown formatting, no code blocks, no explanatory text.

Format:
[
  {{"question": "Your question here", "answer": "The corresponding answer from the context"}},
  {{"question": "Another question", "answer": "Another answer"}}
]

Generate the QA pairs now:"""

    def _parse_qa_pairs_robust(self, generated_text: str, context_id: int) -> List[Dict]:
        """
        Robust parsing of QA pairs using multiple strategies:
        1. Direct JSON parsing.
        2. Pattern-based extraction.
        3. Manual regex extraction as fallback.
        """
        # Strategy 1: Direct JSON parsing
        try:
            qa_pairs = json.loads(generated_text)
            if isinstance(qa_pairs, list) and self._validate_qa_pairs(qa_pairs):
                if self.debug:
                    print(f"[DEBUG] Context {context_id}: Direct JSON parsing successful")
                return qa_pairs
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove code blocks and extract JSON array
        try:
            cleaned = re.sub(r'```(?:json)?\s*', '', generated_text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            json_match = re.search(r'\[[\s\S]*\]', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                qa_pairs = json.loads(json_str)
                if isinstance(qa_pairs, list) and self._validate_qa_pairs(qa_pairs):
                    if self.debug:
                        print(f"[DEBUG] Context {context_id}: JSON array extraction successful")
                    return qa_pairs
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 3: Clean JSON string and attempt parsing
        try:
            cleaned = self._clean_json_string(generated_text)
            qa_pairs = json.loads(cleaned)
            if isinstance(qa_pairs, list) and self._validate_qa_pairs(qa_pairs):
                if self.debug:
                    print(f"[DEBUG] Context {context_id}: Cleaned JSON parsing successful")
                return qa_pairs
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 4: Manual regex extraction
        qa_pairs = self._manual_parse_qa(generated_text)
        if qa_pairs:
            if self.debug:
                print(f"[DEBUG] Context {context_id}: Manual extraction ({len(qa_pairs)} pairs)")
            return qa_pairs

        if self.debug:
            print(f"[DEBUG] Context {context_id}: All parsing strategies failed")
            print(f"[DEBUG] Raw preview: {generated_text[:200]}")
        return []

    def _validate_qa_pairs(self, qa_pairs: List) -> bool:
        """Checks QA pair data validity."""
        if not qa_pairs:
            return False
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                return False
            if 'question' not in pair or 'answer' not in pair:
                return False
            if not isinstance(pair['question'], str) or not isinstance(pair['answer'], str):
                return False
            if not pair['question'].strip() or not pair['answer'].strip():
                return False
        return True

    def _clean_json_string(self, text: str) -> str:
        """Cleans possible code-blocks, control chars from JSON string."""
        text = text.strip()
        text = re.sub(r'```\s*', '', text)
        json_match = re.search(r'\[[\s\S]*\]', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return text

    def _manual_parse_qa(self, text: str) -> List[Dict]:
        """Manual fallback regex-based QA extraction."""
        qa_pairs = []
        pattern1 = r'\{\s*"question"\s*:\s*"([^"]+)"\s*,\s*"answer"\s*:\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern1, text, re.DOTALL)
        for q, a in matches:
            qa_pairs.append({"question": q.strip(), "answer": a.strip()})
        if qa_pairs:
            return qa_pairs
        pattern2 = r'[Qq]uestion\s*:?\s*(.+?)\n\s*[Aa]nswer\s*:?\s*(.+?)(?=\n\s*[Qq]uestion|\n\n|$)'
        matches = re.findall(pattern2, text, re.DOTALL)
        for q, a in matches:
            qa_pairs.append({"question": q.strip(), "answer": a.strip()})
        if qa_pairs:
            return qa_pairs
        pattern3 = r'Q\s*:?\s*(.+?)\n\s*A\s*:?\s*(.+?)(?=\nQ|\n\n|$)'
        matches = re.findall(pattern3, text, re.DOTALL)
        for q, a in matches:
            qa_pairs.append({"question": q.strip(), "answer": a.strip()})
        return qa_pairs

def main():
    parser = argparse.ArgumentParser(
        description="QA Pair Generator for Context Parameterization (via vLLM Server)."
    )
    parser.add_argument(
        "--contexts_path",
        type=str,
        required=True,
        help="Path to contexts JSON file."
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://100.100.0.105:30080/v1",
        help="vLLM server URL."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for vLLM server."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for generation."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=30,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (uses first 5 contexts only, shows detailed logs)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="setting1_qa_pairs.json",
        help="Output file path."
    )

    args = parser.parse_args()

    print("===========================================")
    print("Context QA Pair Generation")
    print("===========================================")
    print(f"Server URL: {args.server_url}")
    print(f"Model: {args.model}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Debug Mode: {args.debug}")
    print(f"Output Path: {args.output}")
    print("===========================================\n")

    generator = ContextQAGenerator(
        base_url=args.server_url,
        api_key=args.api_key,
        model_name=args.model,
        max_workers=args.max_workers,
        debug=args.debug
    )

    try:
        contexts = generator.load_contexts(args.contexts_path)
    except FileNotFoundError:
        print(f"Error: Context file '{args.contexts_path}' not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON at '{args.contexts_path}'!")
        return

    if args.debug:
        contexts = contexts[:5]
        print("DEBUG MODE: Using first 5 contexts only.")

    print(f"{len(contexts)} contexts loaded from {args.contexts_path}\n")

    try:
        generator.generate_qa_pairs(contexts, output_path=args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Partial results may be saved.")
    except Exception as e:
        print(f"\nFatal error: {e}")

if __name__ == "__main__":
    main()