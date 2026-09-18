"""Stateless Snake observations and real MLX action selection."""
from collections import deque
from math import isfinite
from time import perf_counter
from server.snake_planning import enrich, planning_candidates, LOOKAHEAD
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter(prefix='/api/snake', tags=['snake'])
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [('A', 'straight', 0), ('B', 'left', -1), ('C', 'right', 1)]


class SnakeState(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    size: int = Field(default=16, ge=6, le=24)
    snake: list[list[int]] = Field(min_length=1, max_length=576)
    food: list[int] = Field(min_length=2, max_length=2)
    direction: int = Field(ge=0, le=3)
    safety: bool = True
    policy: Literal['classic', 'enhanced', 'hybrid'] = 'enhanced'
    recent_heads: list[list[int]] = Field(default_factory=list, max_length=24)
    steps_since_food: int = Field(default=0, ge=0, le=1000000)

    @model_validator(mode='after')
    def valid_board(self):
        cells = self.snake + [self.food] + self.recent_heads
        if any(len(p) != 2 or any(v < 0 or v >= self.size for v in p) for p in cells):
            raise ValueError('Coordinates must be [x, y] within the board')
        body = [tuple(p) for p in self.snake]
        if len(set(body)) != len(body) or tuple(self.food) in body:
            raise ValueError('Snake and food must occupy distinct cells')
        if any(abs(a[0]-b[0]) + abs(a[1]-b[1]) != 1 for a, b in zip(body, body[1:])):
            raise ValueError('Snake segments must be adjacent')
        if len(body) > 1:
            dx, dy = DIRECTIONS[self.direction]
            if (body[0][0]-body[1][0], body[0][1]-body[1][1]) != (dx, dy):
                raise ValueError('Direction must match head and neck')
        return self


def observe(state: SnakeState) -> list[dict]:
    observations = []
    for label, action, turn in ACTIONS:
        direction = (state.direction + turn) % 4
        dx, dy = DIRECTIONS[direction]
        head = (state.snake[0][0]+dx, state.snake[0][1]+dy)
        eats = head == tuple(state.food)
        # The old tail vacates this tick unless the snake eats.
        blocked = {tuple(p) for p in (state.snake if eats else state.snake[:-1])}
        wall = not (0 <= head[0] < state.size and 0 <= head[1] < state.size)
        collision = wall or head in blocked
        distance = None
        reachable = 0
        if not collision:
            queue = deque([(head, 0)])
            seen = {head}
            while queue:
                point, steps = queue.popleft()
                if point == tuple(state.food):
                    distance = steps
                for sx, sy in DIRECTIONS:
                    nxt = (point[0]+sx, point[1]+sy)
                    if (0 <= nxt[0] < state.size and 0 <= nxt[1] < state.size
                            and nxt not in blocked and nxt not in seen):
                        seen.add(nxt)
                        queue.append((nxt, steps+1))
            reachable = len(seen)
        observations.append(dict(label=label, action=action, direction=direction,
                                 safe=not collision, eats=eats, food_distance=distance,
                                 reachable_cells=reachable))
    return observations


def select_action(observations: list[dict], probabilities: dict[str, float], safety: bool):
    if set(probabilities) != {'A', 'B', 'C'} or any(
        not isfinite(p) or p < 0 or p > 1 for p in probabilities.values()
    ) or abs(sum(probabilities.values()) - 1) > 0.002:
        raise ValueError('Invalid model probability distribution')
    raw = max(observations, key=lambda item: probabilities[item['label']])
    allowed = [o for o in observations if o['safe']] if safety else observations
    if not allowed:
        return raw, None, False
    selected = max(allowed, key=lambda item: probabilities[item['label']])
    return raw, selected, selected['label'] != raw['label']


@router.get('/info')
def info():
    from core.engine import USE_MLX
    return {'backend': 'MLX' if USE_MLX else 'unsupported',
            'model': 'Qwen2.5-1.5B · 4bit', 'actions': ['straight', 'left', 'right']}


@router.post('/decide')
def decide(state: SnakeState):
    started = perf_counter()
    from core.engine import USE_MLX
    if not USE_MLX:
        raise HTTPException(503, 'Snake requires the local MLX backend')
    from core.engine_mlx import get_engine, run_parallel_generation
    from core.schema import StructuredSchema
    observations = observe(state)
    description = ('Choose ONE next move for Snake. A = go straight; B = turn left; '
                   'C = turn right. Avoid collision first; then prefer eating food or '
                   'the shortest reachable route to food. Avoid small enclosed regions.')
    schema = StructuredSchema({'move': {'type': 'enum', 'choices': ['A', 'B', 'C'],
                                        'description': description}})
    _, tokenizer = get_engine()
    if any(len(tokenizer.encode(label, add_special_tokens=False)) != 1 for label in ['A', 'B', 'C']) or any(
        schema.compile_parallel_metadata(tokenizer)['has_collisions']
    ):
        raise HTTPException(503, 'Model tokenizer is incompatible with A/B/C action labels')
    lines = [f'Snake board: {state.size} by {state.size}. Length: {len(state.snake)}.',
             f'Head: {state.snake[0]}. Food: {state.food}.',
             'Compare these next-move observations. Smaller food_distance is better; '
             'UNREACHABLE means no path with the current body held fixed.']
    for obs in observations:
        distance = obs['food_distance'] if obs['food_distance'] is not None else 'UNREACHABLE'
        lines.append(f"{obs['label']} ({obs['action']}): "
                     f"{'SAFE' if obs['safe'] else 'FATAL COLLISION'}, eats_food={obs['eats']}, "
                     f"food_distance={distance}, reachable_cells={obs['reachable_cells']}.")
    if state.policy != 'classic':
        observations = enrich(state, observations)
        description = ('Choose the best next action for Snake: A=straight, B=left, C=right. '
                       'Avoid death, then reach food in the fewest moves. Avoid repeating cells.')
        schema = StructuredSchema({'move': {'type': 'enum', 'choices': ['A', 'B', 'C'],
                                            'description': description}})
        lines = [
            'Choose the shortest safe route to food. Use recent visits only to break ties.',
            f'No food eaten for {state.steps_since_food} moves. Snake length={len(state.snake)}. '
            'Routes simulate tail movement and growth. Escape estimates are not guarantees.',
        ]
        for item in observations:
            prefix = f"{item['label']}. Go {item['action']}. "
            if not item['safe']:
                sentence = 'You will collide immediately. Do not choose this.'
            elif item['survival_steps'] < LOOKAHEAD:
                sentence = f"Forced collision within {LOOKAHEAD} moves. Avoid this trap."
            elif item['route_steps'] is not None and item['escape_after_food']:
                sentence = f"Food is {item['route_steps']} moves away, with an estimated escape after eating."
            elif item['route_steps'] is not None:
                sentence = 'A food route was found, but escape after eating is doubtful.'
            else:
                sentence = 'No food route found within the search budget. '
                sentence += 'Tail access exists.' if item['tail_connected'] else 'Tail access is blocked.'
            lines.append(prefix + sentence + f" Recent visits: {item['recent_visits']}. "
                         f"Available space: {item['reachable_cells']} cells.")
    context = '\n'.join(lines)
    result = run_parallel_generation(context, schema, temperature=1.0)
    probabilities = {item['choice']: item['probability']
                     for item in result['field_telemetry']['move']['top_choices']}
    raw, selected, intervened = select_action(observations, probabilities, state.safety)
    protection_override = intervened
    assistance_reason = None
    if state.policy == 'hybrid' and selected is not None:
        eligible, reason = planning_candidates(observations, state.steps_since_food, state.size)
        if eligible:
            assisted = max(eligible, key=lambda item: probabilities[item['label']])
            if assisted['label'] != selected['label']:
                assistance_reason = reason
            selected = assisted
    planning_override = assistance_reason is not None
    return {'action': selected['action'] if selected else None,
            'direction': selected['direction'] if selected else None,
            'model_action': raw['action'], 'safety_override': protection_override,
            'planning_override': planning_override, 'assistance_reason': assistance_reason,
            'policy': state.policy, 'total_ms': round((perf_counter() - started) * 1000, 2),
            'trapped': selected is None, 'elapsed_ms': result['elapsed_ms'],
            'candidates': [dict(obs, probability=probabilities[obs['label']]) for obs in observations],
            'observation': context}
