"""本地候选决策评分：修改 OPTIONS 和 QUESTION，或从命令行传入问题。"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['MODEL_ID'] = str(ROOT / 'models/qwen2.5-1.5b-instruct-4bit')
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from core.engine_mlx import get_engine, run_parallel_generation
from core.schema import StructuredSchema

# 保留 A/B/C 标签，修改其对应的决策含义。
OPTIONS = {
    'A': '直接提供操作步骤：问题明确且有常规解决办法',
    'B': '转交人工技术支持：已尝试常规办法仍未解决，或需要后台排查',
    'C': '先追问补充信息：描述不足，无法判断具体问题',
}
QUESTION = '用户说：我的软件用不了。应该采取什么下一步行动？'


def decide(question: str, temperature: float = 1.0) -> dict:
    """返回候选项内归一化的模型分数，不是经过校准的正确率。"""
    if not question.strip():
        raise ValueError('问题不能为空')
    if not temperature > 0 or not temperature < float('inf'):
        raise ValueError('temperature 必须是有限正数')
    if not 2 <= len(OPTIONS) <= 5:
        raise ValueError('本示例支持 2 至 5 个选项，避免上游只返回 top 5 的截断')
    _, tokenizer = get_engine()
    tokens = [tokenizer.encode(label, add_special_tokens=False) for label in OPTIONS]
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != len(tokens):
        raise ValueError('请使用 token 各不相同的单 token 标签，如 A/B/C')
    # 上游只使用 description 的第一行，所以映射必须放在同一行。
    descriptions = '; '.join(f'{label} = {text}' for label, text in OPTIONS.items())
    schema = StructuredSchema({'decision': {
        'type': 'enum',
        'choices': list(OPTIONS),
        'description': '请选择最合适的下一步行动，只输出一个标签。' + descriptions,
    }})
    if any(schema.compile_parallel_metadata(tokenizer)['has_collisions']):
        raise ValueError('候选 token 冲突，停止评分，避免进入上游启发式概率分支')
    result = run_parallel_generation(question, schema, temperature=temperature)
    telemetry = result['field_telemetry']['decision']
    scores = telemetry['top_choices']
    assert len(scores) == len(OPTIONS)
    assert abs(sum(item['probability'] for item in scores) - 1.0) < 0.001
    return {
        'question': question,
        'decision': OPTIONS[telemetry['value']],
        'label': telemetry['value'],
        'candidate_probabilities': [
            {'label': item['choice'], 'decision': OPTIONS[item['choice']],
             'probability': item['probability']} for item in scores
        ],
        'temperature': temperature,
        'elapsed_ms': result['elapsed_ms'],
        'note': '候选项内的相对模型分数，未经正确率校准；修改选项、提示词或温度都会影响结果。',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?', default=QUESTION)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(decide(args.question, args.temperature), ensure_ascii=False, indent=2))
