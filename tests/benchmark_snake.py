"""Replay seeded games against real local MLX. Run from the project root."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MODEL_ID'] = str(ROOT / 'models/qwen2.5-1.5b-instruct-4bit')
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from server.snake import SnakeState, decide
from server.snake_planning import step


def run(seed, policy, budget):
    rng = random.Random(seed)
    body, food, direction = ((8, 8), (7, 8), (6, 8)), (11, 8), 1
    heads, hunger, score, revisits, guard, assist = [], 0, 0, 0, 0, 0
    latency, status, count = [], 'step_limit', 0
    for _ in range(budget):
        state = SnakeState(size=16, snake=[list(p) for p in body], food=list(food), direction=direction,
                           policy=policy, recent_heads=heads, steps_since_food=hunger)
        result = decide(state)
        count += 1
        latency.append(result['total_ms'])
        guard += result['safety_override']
        assist += result['planning_override']
        if result['trapped']:
            status = 'trapped'
            break
        direction = result['direction']
        moved = step(body, direction, food, 16)
        if moved is None:
            status = 'collision'
            break
        heads = (heads + [list(body[0])])[-24:]
        body, eats = moved
        revisits += list(body[0]) in heads
        hunger += 1
        if eats:
            score += 1
            hunger = 0
            heads = []  # History is food-target-specific, same as the UI.
            free = [(x, y) for y in range(16) for x in range(16) if (x, y) not in body]
            if not free:
                status = 'won'
                break
            food = rng.choice(free)
    return dict(seed=seed, policy=policy, steps=count, food=score, revisits=revisits,
                safety_overrides=guard, planning_overrides=assist,
                mean_total_ms=round(mean(latency), 2), status=status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--policies', nargs='+', default=['classic', 'enhanced', 'hybrid'])
    parser.add_argument('--output', default='work/snake-benchmark.json')
    args = parser.parse_args()
    results = []
    for seed in range(args.seeds):
        for policy in args.policies:
            row = run(seed, policy, args.steps)
            results.append(row)
            print(json.dumps(row), flush=True)
            output = ROOT / args.output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({'steps_per_game': args.steps, 'results': results}, indent=2)+'\n')
