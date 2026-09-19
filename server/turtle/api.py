import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .stories import STORIES

router = APIRouter(prefix='/api/turtle', tags=['turtle'])
TTL = 7200
LIMIT = 64
LABELS = ['A', 'B', 'C', 'D', 'E']
ANSWERS = {'A':'是', 'B':'否', 'C':'无关', 'D':'请换成一个明确的是非问题', 'E':'汤底未说明，无法确定'}
GUESSES = {'A':'模型判断：核心真相已还原', 'B':'模型判断：方向接近，还缺少关键环节', 'C':'模型判断：还原与汤底有冲突', 'D':'模型判断：还原太模糊，请再具体一点', 'E':'模型无法确定，请继续提问或查看汤底'}


@dataclass
class Game:
    story_id: str
    custom: dict | None = None
    id: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    history: list = field(default_factory=list)
    hints: int = 0
    status: str = 'playing'
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


GAMES = {}
GAMES_LOCK = threading.Lock()


def public(game):
    story=game.custom or STORIES[game.story_id]
    data=dict(id=game.id, story_id=game.story_id, title=story['title'], surface=story['surface'],
              status=game.status, history=game.history, hints_used=game.hints,
              hints_available=len(story['hints']), questions=sum(h['kind']=='question' for h in game.history),
              turns_left=max(0,80-len(game.history)))
    if game.status!='playing':
        data['solution']=story['solution']
    return data


def score(story, history, question, kind):
    from core.engine import USE_MLX
    if not USE_MLX:raise HTTPException(503,'此 Demo 需要本地 MLX 后端')
    from core.engine_mlx import get_engine, run_parallel_generation
    from core.schema import StructuredSchema
    tokens = ['yes','no','other','ask','unknown'] if kind=='question' else LABELS
    label_map = dict(zip(tokens,LABELS))
    description = (
        'Answer the question according to the story facts: yes = factually true; no = contradicted by facts; other = irrelevant; ask = not a clear yes/no question; unknown = not established by facts.'
        if kind=='question' else
        'Judge the proposed solution against ALL required key points. A=SOLVED, all key causal points are correctly explained; B=PARTIAL, some true key points but incomplete; C=WRONG, contradicts the key facts; D=VAGUE, no clear explanation; E=UNCERTAIN, cannot judge reliably.'
    )
    schema=StructuredSchema({'answer':{'type':'enum','choices':tokens,'description':description}})
    _,tokenizer=get_engine()
    if any(len(tokenizer.encode(x,add_special_tokens=False))!=1 for x in tokens) or any(schema.compile_parallel_metadata(tokenizer)['has_collisions']):
        raise HTTPException(503,'候选标签必须是无冲突的单 token')
    import json
    previous=[{'question':h['text'],'answer':h['answer']} for h in history[-6:] if h['kind'] in ('question','guess')]
    context = ('Previous dialogue (may contain mistakes): ' + json.dumps(previous,ensure_ascii=False) + '\n') if previous else ''
    context += 'Story facts: ' + ' '.join(story['facts']) + '\n'
    if kind=='guess':context += 'Required solution points: ' + story['key'] + '\n'
    context += ('Question to judge: ' if kind=='question' else 'Proposed solution to judge: ') + question
    context += '\nAnswer based only on the story facts.'
    result=run_parallel_generation(context,schema,temperature=1.0)
    choices=[dict(choice=label_map[c['choice']],probability=c['probability']) for c in result['field_telemetry']['answer']['top_choices']]
    winner=max(choices,key=lambda c:c['probability'])['choice']
    return winner, choices, result['elapsed_ms']


class NewGame(BaseModel):
    model_config=ConfigDict(extra='forbid')
    story_id: str = 'random'
    title: str = Field(default='',max_length=80)
    surface: str = Field(default='',max_length=1500)
    solution: str = Field(default='',max_length=5000)


class Turn(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind: Literal['question','guess','hint','reveal'] = 'question'
    text: str = Field(default='',max_length=600)


@router.get('/stories')
def stories():
    return [{'id':key,'title':s['title'],'level':s['level'],'surface':s['surface']} for key,s in STORIES.items()]


@router.post('/new')
def new_game(req: NewGame):
    key=secrets.choice(list(STORIES)) if req.story_id=='random' else req.story_id
    custom=None
    if key=='custom':
        if not req.surface.strip() or not req.solution.strip():raise HTTPException(422,'请填写汤面和汤底')
        custom=dict(title=req.title.strip() or '自定义海龟汤',surface=req.surface.strip(),solution=req.solution.strip(),
                    facts=[req.solution.strip()],key='还原应正确解释汤底的主要因果关系：'+req.solution.strip(),hints=[])
    elif key not in STORIES:raise HTTPException(404,'找不到该汤题')
    now=time.monotonic()
    with GAMES_LOCK:
        for gid,g in list(GAMES.items()):
            if now-g.touched>TTL and not g.lock.locked():del GAMES[gid]
        if len(GAMES)>=LIMIT:raise HTTPException(429,'本地会话数量已满，请稍后再试或重启服务')
        game=Game(key,custom=custom);GAMES[game.id]=game
    return public(game)


@router.post('/{game_id}/turn')
def turn(game_id: str, req: Turn):
    with GAMES_LOCK:game=GAMES.get(game_id)
    if game is None or time.monotonic()-game.touched>TTL:raise HTTPException(404,'本局已失效，请开一局新游戏')
    if not game.lock.acquire(blocking=False):raise HTTPException(409,'主持人正在回答，请稍候')
    try:
        game.touched=time.monotonic()
        if game.status!='playing':raise HTTPException(409,'本局已经结束，请开始新的一局')
        story=game.custom or STORIES[game.story_id]
        if req.kind=='reveal':
            game.status='revealed'
            return public(game)
        if len(game.history)>=80:raise HTTPException(409,'本局已达到80轮上限，可以查看汤底或重新开始')
        if req.kind=='hint':
            if game.hints>=len(story['hints']):raise HTTPException(409,'提示已全部用完')
            answer=story['hints'][game.hints];game.hints+=1
            entry=dict(kind='hint',text='请求提示',answer=answer,scores=[],elapsed_ms=0)
        else:
            text=req.text.strip()
            if not text:raise HTTPException(422,'请输入问题或还原内容')
            winner,scores,elapsed=score(story,game.history,text,req.kind)
            wording=ANSWERS if req.kind=='question' else GUESSES
            if winner not in wording:raise HTTPException(503,'模型返回了无效标签')
            entry=dict(kind=req.kind,text=text,answer=wording[winner],label=winner,
                       scores=[dict(label=c['choice'],name=wording[c['choice']],probability=c['probability']) for c in scores],elapsed_ms=elapsed)
            if req.kind=='guess' and winner=='A':game.status='solved'
        game.history.append(entry)
        return public(game)
    finally:game.lock.release()
