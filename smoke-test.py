"""在普通 macOS 终端运行：.venv/bin/python smoke-test.py"""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ['MODEL_ID'] = str(root / 'models/qwen2.5-1.5b-instruct-4bit')
os.environ['HF_HOME'] = str(root / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import mlx.core as mx
from core.engine_mlx import run_parallel_generation
from core.schema import StructuredSchema

schema = StructuredSchema({
    'sentiment': {'type': 'enum', 'choices': ['POSITIVE', 'NEGATIVE'],
                  'description': 'Sentiment expressed in the customer review'},
    'mentions_delivery': {'type': 'boolean', 'description': 'Whether the review mentions delivery'}
})
result = run_parallel_generation('Great product! Fast delivery. I love it.', schema)
assert result['is_valid_json'] and result['schema_match'], result
print(json.dumps(result, ensure_ascii=False, indent=2))
