export const VECTORS = [[0,-1],[1,0],[0,1],[-1,0]];
export function spawnFood(snake, size, random = Math.random) {
  const occupied = new Set(snake.map(p => p.join(',')));
  const free = [];
  for (let y=0; y<size; y++) for (let x=0; x<size; x++)
    if (!occupied.has(`${x},${y}`)) free.push([x,y]);
  return free.length ? free[Math.floor(random()*free.length)] : null;
}
export function newGame(size=16) {
  const c = Math.floor(size/2);
  return {size, snake:[[c,c],[c-1,c],[c-2,c]], food:[c+3,c], direction:1,
          score:0, steps:0, recentHeads:[], stepsSinceFood:0, status:'ready'};
}
export function advance(state, direction, random=Math.random) {
  if (state.status === 'dead' || state.status === 'won') return state;
  if (!Number.isInteger(direction) || direction < 0 || direction > 3) throw new Error('Invalid direction');
  if ((direction+2)%4 === state.direction) direction = state.direction;
  const [dx,dy] = VECTORS[direction];
  const head = [state.snake[0][0]+dx,state.snake[0][1]+dy];
  const eats = head[0]===state.food[0] && head[1]===state.food[1];
  const body = eats ? state.snake : state.snake.slice(0,-1);
  if (head.some(v=>v<0 || v>=state.size) || body.some(p=>p[0]===head[0] && p[1]===head[1]))
    return {...state, status:'dead', steps:state.steps+1};
  const snake = [head,...body];
  const food = eats ? spawnFood(snake,state.size,random) : state.food;
  const recentHeads = eats ? [] : [...(state.recentHeads ?? []),state.snake[0]].slice(-24);
  return {...state,snake,food,direction,recentHeads,stepsSinceFood:eats?0:(state.stepsSinceFood??0)+1,score:state.score+(eats?1:0),steps:state.steps+1,
          status:food ? 'playing' : 'won'};
}
