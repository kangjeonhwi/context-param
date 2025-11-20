# src/adapter_scanner.py
import os
from pathlib import Path
from typing import List, Dict, Any

class AdapterScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def scan_sample_adapters(self, sample_id: int) -> List[Dict[str, str]]:
        """
        특정 sample_id 폴더 하위의 모든 유효한 adapter 경로를 탐색합니다.
        """
        sample_dir = self.root_dir / f"sample{sample_id}"
        if not sample_dir.exists():
            print(f"Warning: Directory for sample {sample_id} not found at {sample_dir}")
            return []

        adapters = []
        for root, dirs, files in os.walk(sample_dir):
            if "adapter_config.json" in files:
                abs_path = Path(root)
                
                try:
                    rel_path = abs_path.relative_to(sample_dir)
                    if rel_path.name == 'adapter':
                        config_name = str(rel_path.parent)
                    else:
                        config_name = str(rel_path)
                except ValueError:
                    config_name = str(abs_path.name)

                adapters.append({
                    "id": config_name, 
                    "path": str(abs_path),
                    "type": "dora" if "dora" in config_name else "lora"
                })
        
        adapters.sort(key=lambda x: x['id'])
        return adapters

    def get_debug_adapters(self, sample_id: int, limit: int = 2):
        adapters = self.scan_sample_adapters(sample_id)
        return adapters[:limit]