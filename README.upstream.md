---
language:
- en
license: apache-2.0
library_name: mlx
tags:
- structured-generation
- parallel-decoding
- constrained-decoding
- apple-silicon
- mlx
- classification
- json
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
spaces:
- drinkmoonshine/parallel-constrained-decoding
---

# Parallel Constrained Decoding for Apple Silicon

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/drinkmoonshine/parallel-constrained-decoding)

> **Live Demo**: Try the side-by-side comparison live on Hugging Face Spaces: [drinkmoonshine/parallel-constrained-decoding](https://huggingface.co/spaces/drinkmoonshine/parallel-constrained-decoding).

A high-throughput inference engine for structured information extraction, decision routing, and categorical classification on Apple Silicon using MLX.

Parallel Constrained Decoding evaluates multi-field JSON schemas simultaneously rather than generating tokens sequentially. On an Apple Silicon M4 Max, it delivers **5.6x to 7.0x latency reductions** compared to standard autoregressive decoding with **100% schema validity** and **calibrated field-level confidence scores**.

---

## Performance Benchmarks (Apple Silicon M4 Max)

Evaluated with `mlx-community/Qwen2.5-1.5B-Instruct-4bit` on macOS Sequoia:

| Scenario | Fields | Autoregressive Baseline | Parallel Constrained | Latency Speedup | Syntax Validity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fintech Fraud Routing** | 4 fields | 420 ms (120 tok/s) | **75 ms** | **5.6x** | 100% guaranteed |
| **Code Security Audit** | 4 fields | 380 ms (125 tok/s) | **68 ms** | **5.6x** | 100% guaranteed |
| **High-Cardinality Tariff** | 1 field (255 choices) | 500 ms (118 tok/s) | **89 ms** | **5.6x** | 100% guaranteed |
| **Enterprise Support Triage** | 28 fields | 1,900 ms (130 tok/s) | **270 ms** | **7.0x** | 100% guaranteed |

---

## Why Parallel Constrained Decoding?

### The Problem with Autoregressive Structured Generation

Standard LLM structured generation (such as JSON mode or grammar-guided sampling) relies on token-by-token autoregressive decoding:

```
[Context Prompt] -> "{" -> "\n" -> " " -> "risk" -> ":" -> " " -> "HIGH" -> ...
(Requires 150 to 500 sequential forward passes)
```

Each token requires a distinct GPU/NPU forward pass and sequential memory bandwidth roundtrips. As schema size grows, latency scales linearly with output token length:

$$T_{\text{autoregressive}} = \sum_{k=1}^{K} t_{\text{step}}(k)$$

Additionally, autoregressive decoding is susceptible to syntax degradation, field omission, and hallucinated keys.

### The Solution: Parallel Evaluation via KV-Cache Broadcasting

In structured extraction and classification, field values belong to bounded candidate sets (booleans or categorical enums). Parallel Constrained Decoding exploits this property:

```
                          +---> [Field 1: "risk_level"] -------> Logit Slicing -> Top Choice
                          |
[Context Prefix Prefill] -+---> [Field 2: "requires_review"] ---> Logit Slicing -> Top Choice
(Single KV-Cache State)   |
                          +---> [Field M: "action_tier"] ------> Logit Slicing -> Top Choice
                          
                     (All fields evaluated simultaneously)
```

1. **Single Broadcast Prefill**: The context document and semantic schema descriptions are prefilled once into an MLX Key-Value (KV) cache.
2. **KV-Cache Broadcasting**: The KV-cache is broadcast across all $M$ schema fields in parallel.
3. **Sub-Vocabulary Logit Slicing**: For each field, only candidate token IDs belonging to valid schema choices are evaluated. The remaining vocabulary is masked.
4. **Calibrated Softmax Probabilities**: Exact normalized probabilities are calculated over the candidate slice:
   $$P(c_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{C} \exp(z_j / T)}$$
5. **Token Tree Disambiguation**: When candidate choices share multi-token prefix roots, the engine executes continuation steps using sliced cache states with zero memory reallocation.
6. **Programmatic Assembly**: Output JSON is constructed directly from verified values, guaranteeing 100% valid syntax without JSON parsing errors.

---

## Installation

### Prerequisites

- Apple Silicon Mac (M1, M2, M3, M4 series)
- macOS 14.0 or later
- Python 3.10+

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/parallel-constrained-decoding.git
cd parallel-constrained-decoding

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Developer SDK Quickstart

### 1. Defining Schemas

Schemas are defined using `StructuredSchema`. Each field specifies a `type` (`enum` or `boolean`), a `description` to guide model reasoning, and `choices` (for enum types, supporting up to 255 choices):

```python
from core.schema import StructuredSchema, FieldDefinition

# Option A: Dictionary-based definition
schema_dict = {
    "priority": {
        "type": "enum",
        "choices": ["P0_CRITICAL", "P1_HIGH", "P2_NORMAL", "P3_LOW"],
        "description": "Urgency tier based on customer business impact"
    },
    "requires_escalation": {
        "type": "boolean",
        "description": "Whether an on-call engineer must be notified immediately"
    },
    "department": {
        "type": "enum",
        "choices": ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"],
        "description": "Target handling department"
    }
}

schema = StructuredSchema(schema_dict)
```

You can also construct fields explicitly using `FieldDefinition`:

```python
fields = {
    "tariff_classification": FieldDefinition(
        name="tariff_classification",
        field_type="enum",
        description="Harmonized System 6-digit tariff category code",
        choices=["0101.21", "0101.29", "8471.30", "8517.12", "8542.31", ...] # Up to 255 choices
    )
}
```

### 2. Running Parallel Generation

Execute parallel constrained inference on your context string:

```python
from core.engine import run_parallel_generation

context = """
Incident Report: Production database db-primary-01 CPU at 100%.
Payment gateway failing for 40% of checkout requests.
Tier 1 Enterprise customer affected: Acme Global.
"""

result = run_parallel_generation(context, schema)

print(f"Latency: {result['elapsed_ms']} ms")
print(f"Prefill Time: {result['prefill_ms']} ms")
print(f"Passes: {result['sequential_forward_passes']}")
print("\nExtracted JSON:")
print(result["parsed_json"])
```

### 3. Response Structure

The output dictionary provides both the structured JSON and detailed field telemetry:

```python
{
    "mode": "parallel_constrained_calibrated",
    "elapsed_ms": 74.5,
    "prefill_ms": 52.1,
    "suffix_eval_ms": 18.2,
    "sequential_forward_passes": 1,
    "is_valid_json": True,
    "schema_match": True,
    "parsed_json": {
        "priority": { "value": "P0_CRITICAL", "prob": 0.9924 },
        "requires_escalation": { "value": "true", "prob": 0.9981 },
        "department": { "value": "INFRASTRUCTURE", "prob": 0.9815 }
    },
    "field_telemetry": {
        "priority": {
            "value": "P0_CRITICAL",
            "confidence": 0.9924,
            "cardinality": 4,
            "top_choices": [
                { "choice": "P0_CRITICAL", "probability": 0.9924 },
                { "choice": "P1_HIGH", "probability": 0.0068 },
                { "choice": "P2_NORMAL", "probability": 0.0006 },
                { "choice": "P3_LOW", "probability": 0.0002 }
            ]
        }
    }
}
```

### 4. Streaming Autoregressive Baseline

To compare against standard autoregressive generation:

```python
from core.engine import stream_naive_generation

for event in stream_naive_generation(context, schema):
    if event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "done":
        print(f"\nCompleted in {event['result']['elapsed_ms']} ms")
```

---

## Interactive Web Visualizer

The repository includes a web interface for side-by-side latency and accuracy comparison.

To launch the web server:

```bash
bash run.sh
```

Or run directly with uvicorn:

```bash
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Features

- **Side-by-Side Comparison**: Parallel Constrained Decoding vs. Autoregressive Streaming.
- **Live Millisecond Timers**: Real-time elapsed latency counters.
- **Synchronized Scrolling**: Matching keys align across both panes.
- **Interactive Row Highlighting**: Hover over any field in either panel to highlight the corresponding key in the other.
- **Hallucination Detection**: Highlights omitted or hallucinated keys in naive autoregressive output.

---

## Command-Line Benchmark Runner

Run the benchmark suite across pre-configured enterprise presets:

```bash
python3 -m core.benchmark
```

Output example:

```text
======================================================================
Parallel Constrained vs. Autoregressive Generation Benchmark
======================================================================
--> Running preset: Fintech Fraud Detection (4 fields)...
    Autoregressive Baseline :    421.3 ms | 148 tokens (122.4 tok/s) | Passes: 148
    Parallel Constrained    :     74.8 ms |   0 tokens (O(1))           | Passes: 1
    >> SPEEDUP: 5.6x faster (Step reduction: 148.0x)
    >> Schema match: Naive=True | Parallel=True (100% guaranteed)
----------------------------------------------------------------------
--> Running preset: Support Triage Matrix (28 fields)...
    Autoregressive Baseline :   1894.2 ms | 312 tokens (131.2 tok/s) | Passes: 312
    Parallel Constrained    :    268.4 ms |   0 tokens (O(1))           | Passes: 1
    >> SPEEDUP: 7.1x faster (Step reduction: 312.0x)
    >> Schema match: Naive=True | Parallel=True (100% guaranteed)
----------------------------------------------------------------------
--> Running preset: High-Cardinality Tariff (1 field, 255 choices)...
    Autoregressive Baseline :    498.7 ms |  42 tokens (116.5 tok/s) | Passes: 42
    Parallel Constrained    :     88.6 ms |   0 tokens (O(1))           | Passes: 1
    >> SPEEDUP: 5.6x faster (Step reduction: 42.0x)
    >> Schema match: Naive=True | Parallel=True (100% guaranteed)
----------------------------------------------------------------------
```

---

## Repository Structure

```text
.
├── core/
│   ├── __init__.py           # SDK package exports
│   ├── engine.py             # Parallel constrained decoding & autoregressive engines
│   ├── schema.py             # Schema definitions, metadata compiler & logit mapping
│   ├── prompt_builder.py     # Prompt templates for prefill catalog and naive baseline
│   └── benchmark.py          # Command-line benchmark runner
├── presets/
│   ├── fintech_fraud.json    # Fraud detection scenario (4 fields)
│   ├── code_security.json    # Vulnerability audit scenario (4 fields)
│   ├── support_triage.json   # Enterprise ticket triage (28 fields)
│   └── high_cardinality_255.json # 255-choice tariff classifier
├── server/
│   ├── app.py                # FastAPI endpoints (/api/run-parallel, /api/stream-naive)
│   └── main.py               # Server launcher
├── web/
│   ├── index.html            # Side-by-side comparison UI
│   ├── app.js                # Frontend streaming & synchronized scrolling
│   └── style.css             # UI styling
├── MODEL_CARD.md             # Hugging Face model card documentation
├── requirements.txt          # Python package requirements
├── run.sh                    # Startup script
└── README.md                 # Project documentation
```

---

## Supported Models

The engine is currently configured for `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

Any decoder LLM supported by `mlx-lm` can be loaded by setting `MODEL_ID` in `core/engine.py`.

---

## License

Apache 2.0
