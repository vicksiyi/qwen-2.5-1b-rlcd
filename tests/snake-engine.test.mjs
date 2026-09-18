import test from 'node:test';
import assert from 'node:assert/strict';
import {newGame, advance, spawnFood} from '../web/snake-engine.mjs';
test('food grows snake and never respawns on body',()=>{
  let state=newGame(); state.food=[9,8]; const next=advance(state,1,()=>0);
  assert.equal(next.score,1); assert.equal(next.snake.length,4);
  assert.ok(!next.snake.some(p=>p.join()===next.food.join())); assert.equal(state.snake.length,3);
});
test('reversing is ignored',()=>{const s=advance(newGame(),3);assert.deepEqual(s.snake[0],[9,8]);assert.equal(s.direction,1);});
test('wall is terminal',()=>{let s=newGame();s.snake=[[15,8],[14,8],[13,8]];s=advance(s,1);assert.equal(s.status,'dead');assert.equal(advance(s,0),s);});
test('can enter vacating tail but not body',()=>{
  let s={...newGame(),snake:[[2,2],[2,3],[1,3],[1,2]],direction:0};
  assert.equal(advance(s,3).status,'playing');s.snake.push([1,1]);assert.equal(advance(s,3).status,'dead');
});
test('full board wins without food-spawn loop',()=>{
  const s={size:2,snake:[[0,0],[0,1],[1,1]],food:[1,0],direction:0,score:0,steps:0,status:'playing'};
  const next=advance(s,1);assert.equal(next.status,'won');assert.equal(next.food,null);assert.equal(spawnFood(next.snake,2),null);
});
test('memory is bounded, target-specific and reset on eating',()=>{
  let s=newGame();s.recentHeads=Array.from({length:24},()=>[0,0]);s.stepsSinceFood=10;
  let next=advance(s,0);assert.equal(next.recentHeads.length,24);assert.deepEqual(next.recentHeads.at(-1),[8,8]);assert.equal(next.stepsSinceFood,11);
  s.food=[9,8];next=advance(s,1);assert.deepEqual(next.recentHeads,[]);assert.equal(next.stepsSinceFood,0);
  assert.deepEqual(newGame().recentHeads,[]);
});
