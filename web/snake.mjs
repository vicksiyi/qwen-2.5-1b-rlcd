import {newGame, advance, VECTORS} from './snake-engine.mjs';
const $ = id => document.getElementById(id);
const actions = {straight:{name:'直行',icon:'↑'},left:{name:'左转',icon:'↰'},right:{name:'右转',icon:'↱'}};
let game = newGame(), mode = 'ai', running = false, busy = false, connected = false;
let generation = 0, timer = null, queuedDirection = null, overrides = 0, planningOverrides = 0, history = [];
let best = 0;
try { best = Number(localStorage.getItem('snake-lab-best')) || 0; } catch {}
const ctx = $('board').getContext('2d');
$('probabilities').innerHTML = Object.entries(actions).map(([key,item])=>`<div class="prob-row" id="prob-${key}"><div class="prob-label"><span>${item.icon} ${item.name}<small id="hint-${key}"></small></span><b id="value-${key}">—</b></div><div class="prob-track"><div class="prob-fill" id="fill-${key}"></div></div></div>`).join('');
function draw() {
  const canvas=$('board'), n=game.size, cell=canvas.width/n;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#101811'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#213021';ctx.lineWidth=1;
  for(let i=1;i<n;i++){ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,canvas.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cell);ctx.lineTo(canvas.width,i*cell);ctx.stroke();}
  if(game.food){const [x,y]=game.food;ctx.shadowColor='#f3ad7360';ctx.shadowBlur=24;ctx.fillStyle='#f3ad73';ctx.beginPath();ctx.roundRect(x*cell+cell*.26,y*cell+cell*.26,cell*.48,cell*.48,cell*.15);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#ffcf9e';ctx.fillRect(x*cell+cell*.37,y*cell+cell*.3,cell*.12,cell*.08);}
  // Draw from tail to head, with connected segments and a distinct head.
  for(let i=game.snake.length-1;i>=0;i--){const [x,y]=game.snake[i];ctx.fillStyle=i===0?'#d5fa99':`hsl(91 44% ${Math.max(33,63-i*.8)}%)`;ctx.beginPath();ctx.roundRect(x*cell+5,y*cell+5,cell-10,cell-10,9);ctx.fill();if(i>0){const [px,py]=game.snake[i-1];ctx.beginPath();ctx.roundRect(Math.min(x,px)*cell+cell*.3,Math.min(y,py)*cell+cell*.3,Math.abs(x-px)*cell+cell*.4,Math.abs(y-py)*cell+cell*.4,3);ctx.fill();}}
  const [hx,hy]=game.snake[0], [dx,dy]=VECTORS[game.direction];
  for(const side of [-1,1]){ctx.fillStyle='#203019';ctx.beginPath();ctx.arc((hx+.5)*cell+dx*cell*.19+dy*side*cell*.16,(hy+.5)*cell+dy*cell*.19-dx*side*cell*.16,3.6,0,Math.PI*2);ctx.fill();}
  $('score').textContent=String(game.score).padStart(2,'0');$('steps').textContent=String(game.steps).padStart(2,'0');$('best').textContent=String(best).padStart(2,'0');
}
function setStatus(message){$('status-message').textContent=message;}
function updateControls(){
  $('mode-ai').classList.toggle('active',mode==='ai');$('mode-manual').classList.toggle('active',mode==='manual');
  $('mode-ai').setAttribute('aria-pressed',String(mode==='ai'));$('mode-manual').setAttribute('aria-pressed',String(mode==='manual'));
  $('safety').disabled=mode==='manual'||$('policy').value==='hybrid';
  $('policy').disabled=mode==='manual';
  document.querySelector('.touch-controls').classList.toggle('visible',mode==='manual');
  const text=running?'暂停游戏':(game.status==='dead'||game.status==='won'?'再来一局':game.status==='ready'?'开始游戏':'继续游戏');
  $('toggle').textContent=text+(running?' Ⅱ':' ↗');
  $('play-state').textContent=running?(mode==='ai'?'模型驾驶中':'手动操作中'):(game.status==='dead'?'本局结束':game.status==='won'?'棋盘已填满':game.status==='ready'?'准备就绪':'已暂停');
  $('keyboard-hint').textContent=mode==='manual'?'方向键 / WASD · SPACE 暂停':'SPACE 暂停 · R 重新开始';
}
function overlay(title,text,button='继续游戏'){
  $('overlay').classList.remove('hidden');$('overlay-title').textContent=title;$('overlay-text').textContent=text;$('overlay-start').textContent=button+' ↗';
}
function pause(message='暂停一下，下一步不急。'){
  running=false;generation++;clearTimeout(timer);updateControls();overlay('已暂停',message);setStatus(message);
}
function reset(){
  running=false;generation++;clearTimeout(timer);game=newGame();queuedDirection=null;overrides=0;planningOverrides=0;history=[];
  $('history').replaceChildren();const empty=document.createElement('span');empty.className='history-empty';empty.textContent='每一个选择，都在这里留下轨迹。';$('history').append(empty);
  $('override-count').textContent='防撞 0 次 · 规划 0 次';$('latency').textContent='— ms';$('observation').textContent='开始模型驾驶后显示。';
  for(const action of Object.keys(actions)){ $('prob-'+action).className='prob-row';$('value-'+action).textContent='—';$('hint-'+action).textContent='';$('fill-'+action).style.width='0%'; }
  decisionNote(mode==='ai'?'等待第一步决策':'当前为手动操作',mode==='ai'?'每一步均调用本机模型':'切换到模型驾驶后显示动作概率。');draw();updateControls();overlay('新的一局，新的选择','选择模型驾驶，或亲自掌控方向。','开始游戏');setStatus('准备好后，看看模型会怎么走。');
}
function start(){
  if(mode==='ai'&&!connected){setStatus('本地模型尚未就绪。可以先选择手动操作。');return;}
  if(game.status==='dead'||game.status==='won')reset();
  running=true;game.status='playing';$('overlay').classList.add('hidden');updateControls();schedule(0);
}
function schedule(delay){clearTimeout(timer);if(running)timer=setTimeout(tick,delay);}
function decisionNote(title,note){$('decision-note').replaceChildren(document.createTextNode(title));const span=document.createElement('span');span.textContent=note;$('decision-note').append(span);}
function showDecision(result){
  for(const item of result.candidates){
    $('prob-'+item.action).className='prob-row'+(item.action===result.action?' selected':'')+(!item.safe?' unsafe':'');
    $('value-'+item.action).textContent=(item.probability*100).toFixed(1)+'%';$('fill-'+item.action).style.width=(item.probability*100)+'%';
    $('hint-'+item.action).textContent=!item.safe?'下一步碰撞':item.survival_steps<6?'几步内困死':item.escape_after_food===false?'进食后难脱身':item.eats?'可以吃到食物':item.recent_visits>0?'重复访问 '+item.recent_visits+' 次':'';
  }
  $('latency').textContent=Math.round(result.total_ms??result.elapsed_ms)+' ms';$('observation').textContent=result.observation;
  if(result.trapped){decisionNote('已无安全动作','模型已评分，但三个方向都将碰撞。');return;}
  const chosen=actions[result.action].name;
  if(result.safety_override)overrides++;
  if(result.planning_override){
    planningOverrides++;
    const reasons={safe_food_route:'优先选择预测可脱身的短路径',break_loop:'减少重复绕圈',survival_forecast:'避开预测死路'};
    decisionNote('执行 '+chosen+' · 规划辅助介入','模型首选 '+actions[result.model_action].name+'；'+(reasons[result.assistance_reason]??'依据预测调整动作')+'。上方仍显示模型原始分数。');
  }else if(result.safety_override){decisionNote('执行 '+chosen+' · 防撞保护介入','模型原选 '+actions[result.model_action].name+'；改用安全动作中模型评分最高的一项。');}
  else decisionNote('模型选择 · '+chosen,result.policy==='hybrid'?'本步模型首选与规划建议一致。':$('safety').checked?'本步未改变模型首选。':'模型原始决策 · 防撞保护关闭');
  $('override-count').textContent='防撞 '+overrides+' 次 · 规划 '+planningOverrides+' 次';
}
function addHistory(action,guarded=false){
  history.unshift({action,guarded,step:game.steps});history=history.slice(0,12);$('history').replaceChildren();
  for(const item of history){const chip=document.createElement('div');chip.className='move-chip'+(item.guarded?' guarded':'');chip.title=actions[item.action].name+(item.guarded?' · 辅助介入':'');const icon=document.createElement('b');icon.textContent=actions[item.action].icon;const step=document.createElement('span');step.textContent='#'+String(item.step).padStart(3,'0');chip.append(icon,step);$('history').append(chip);}
}
function finish(){
  running=false;generation++;clearTimeout(timer);updateControls();
  overlay(game.status==='won'?'你填满了整个棋盘！':'这一局，到这里。',`吃到 ${game.score} 个食物，走了 ${game.steps} 步。调整控制方式，再试一次。`,'再来一局');
  setStatus(game.status==='won'?'完美收官。':'本局结束，可以重新开始。');
}
async function tick(){
  if(!running||busy)return;
  busy=true;const epoch=generation, started=performance.now();let result=null;
  try{
    let direction=queuedDirection??game.direction;queuedDirection=null;
    if(mode==='ai'){
      setStatus('模型正在观察棋盘并选择下一步…');
      const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),30000);
      let response;
      try{response=await fetch('/api/snake/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({size:game.size,snake:game.snake,food:game.food,direction:game.direction,safety:$('safety').checked,policy:$('policy').value,recent_heads:game.recentHeads,steps_since_food:game.stepsSinceFood}),signal:controller.signal});}finally{clearTimeout(timeout);}
      if(!response.ok){const err=await response.json().catch(()=>({}));throw new Error(typeof err.detail==='string'?err.detail:'模型请求失败（'+response.status+'）');}
      result=await response.json();if(epoch!==generation||!running)return;
      showDecision(result);
      if(result.trapped){game.status='dead';finish();return;}
      direction=result.direction;
    }
    if(epoch!==generation||!running)return;
    const prior=game.direction;game=advance(game,direction);
    if(game.score>best){best=game.score;try{localStorage.setItem('snake-lab-best',String(best));}catch{}}
    const turn=(direction-prior+4)%4;addHistory(result?.action??(turn===3?'left':turn===1?'right':'straight'),Boolean(result?.safety_override||result?.planning_override));
    draw();if(game.status==='dead'||game.status==='won'){finish();return;}
    setStatus(mode==='ai'?`已执行 ${actions[result.action].name} · ${Math.round(result.elapsed_ms)} ms 推理 · ${game.stepsSinceFood} 步未进食`:'用方向键或 WASD，带它找到下一口。');
  }catch(error){if(epoch===generation&&running){pause(error.name==='AbortError'?'模型响应超时，已暂停。点击继续重试。':'推理出错：'+error.message);}}
  finally{busy=false;if(running)schedule(Math.max(20,Number($('speed').value)-(performance.now()-started)));}
}
function manualDirection(direction){
  if(mode!=='manual')return;
  if((direction+2)%4!==game.direction)queuedDirection=direction;
}
$('toggle').onclick=()=>running?pause():start();$('overlay-start').onclick=start;$('reset').onclick=reset;
for(const selected of ['ai','manual'])$('mode-'+selected).onclick=()=>{
  if(mode===selected)return;const wasRunning=running;if(wasRunning)pause('控制方式已切换。');mode=selected;queuedDirection=null;
  updateControls();setStatus(mode==='manual'?'使用方向键 / WASD，也可以点击方向按钮。':'每一步由本机模型选择。');if(mode==='manual')decisionNote('当前为手动操作','概率面板保留最近一次模型评分。');if(wasRunning)start();
};
$('policy').onchange=()=>{
  generation++;
  const policy=$('policy').value;
  if(policy==='hybrid')$('safety').checked=true;
  const notes={classic:'原始观测和提示词，用于对照。',enhanced:'仅增加记忆与预测，不做规划接管；小模型仍可能绕圈。',hybrid:'规划筛选可脱身的短路径，模型在候选动作中选择；介入会标记。'};
  $('policy-description').textContent=notes[policy];updateControls();
  setStatus('策略已切换，下一步生效。');
};
$('speed').oninput=()=>{$('speed-label').textContent=$('speed').value+' ms';};
$('safety').onchange=()=>{generation++;setStatus($('safety').checked?'防撞保护已开启。':'防撞保护已关闭，下一步执行模型原始首选。');};
document.querySelectorAll('[data-dir]').forEach(button=>button.onclick=()=>manualDirection(Number(button.dataset.dir)));
document.addEventListener('keydown',event=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;
  const map={ArrowUp:0,ArrowRight:1,ArrowDown:2,ArrowLeft:3,w:0,d:1,s:2,a:3};const key=event.key.length===1?event.key.toLowerCase():event.key;
  if(key in map){event.preventDefault();manualDirection(map[key]);}
  if(event.code==='Space'&&document.activeElement.tagName!=='BUTTON'){event.preventDefault();if(!event.repeat)(running?pause():start());}
  if(key==='r'&&!event.repeat)reset();
});
async function connect(){
  try{const response=await fetch('/api/snake/info');if(!response.ok)throw new Error();const data=await response.json();connected=data.backend==='MLX';$('connection').textContent=connected?'本地模型已就绪':'需要 MLX 后端';$('connection-dot').className='dot '+(connected?'green':'red');}
  catch{connected=false;$('connection').textContent='本地服务未连接';$('connection-dot').className='dot red';}
  if(!connected)setTimeout(connect,5000);
}
draw();updateControls();connect();
