import json
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.turtle.api import router, GAMES


class TurtleTests(unittest.TestCase):
    def setUp(self):
        GAMES.clear();app=FastAPI();app.include_router(router);self.client=TestClient(app)
        completion_patch=patch('server.turtle.api.check_completion',return_value=(False,10))
        self.completion=completion_patch.start();self.addCleanup(completion_patch.stop)
    def start(self,**payload):return self.client.post('/api/turtle/new',json=payload).json()
    def turn(self,game,**payload):return self.client.post('/api/turtle/'+game['id']+'/turn',json=payload)
    def test_custom_secret_not_in_public_payload(self):
        secret='只有汤底存在的独特秘密XYZ'
        game=self.start(story_id='custom',surface='为什么呢？',solution=secret,title='我的题目')
        self.assertEqual(game['title'],'我的题目');self.assertNotIn(secret,json.dumps(game,ensure_ascii=False))
        self.assertNotIn('solution',game);self.assertEqual(game['hints_available'],0)
        with patch('server.turtle.api.score',return_value=('A',[{'choice':'A','probability':1.0}],12)) as score:
            result=self.turn(game,kind='question',text='这是问题吗？').json()
            self.assertNotIn(secret,json.dumps(result,ensure_ascii=False))
            self.assertEqual(score.call_args.args[0]['solution'],secret)
        result=self.turn(game,kind='reveal').json();self.assertEqual(result['solution'],secret)
        self.assertEqual(self.turn(game,kind='question',text='继续？').status_code,409)
    def test_story_list_has_no_solution(self):
        stories=self.client.get('/api/turtle/stories').json()
        self.assertEqual(len(stories),3)
        for story in stories:self.assertEqual(set(story),{'id','title','level','surface'})
    def test_validation_and_session(self):
        self.assertEqual(self.client.post('/api/turtle/new',json={'story_id':'custom','surface':'x','solution':' '}).status_code,422)
        self.assertEqual(self.client.post('/api/turtle/new',json={'story_id':'missing'}).status_code,404)
        self.assertEqual(self.client.post('/api/turtle/bad/turn',json={'kind':'hint'}).status_code,404)
        game=self.start(story_id='blank')
        self.assertEqual(self.turn(game,kind='question',text=' ').status_code,422)
    def test_hints_are_bounded_and_win_reveals(self):
        game=self.start(story_id='blank')
        for i in range(3):
            result=self.turn(game,kind='hint').json();self.assertEqual(result['hints_used'],i+1);self.assertNotIn('solution',result)
        self.assertEqual(self.turn(game,kind='hint').status_code,409)
        with patch('server.turtle.api.score',return_value=('B',[{'choice':'B','probability':1}],1)):
            result=self.turn(game,kind='guess',text='部分猜测').json();self.assertEqual(result['status'],'playing');self.assertNotIn('solution',result)
        self.completion.return_value=(True,10)
        with patch('server.turtle.api.score',return_value=('A',[{'choice':'A','probability':1}],1)):
            result=self.turn(game,kind='guess',text='完整猜测').json();self.assertEqual(result['status'],'solved');self.assertIn('solution',result)
    def test_normal_question_can_complete_and_reveal(self):
        game=self.start(story_id='custom',surface='为什么纸上没有字却满分？',solution='盲文凸点答案全对，老师通过触摸批改。')
        self.completion.return_value=(True,8)
        with patch('server.turtle.api.score',return_value=('A',[{'choice':'A','probability':1}],12)):
            result=self.turn(game,kind='question',text='学生用盲文凸点作答，老师触摸读出正确答案，所以给满分，对吗？').json()
        self.assertEqual(result['status'],'solved')
        self.assertIn('已经完全猜出答案',result['history'][-1]['answer'])
        self.assertTrue(result['history'][-1]['completed'])
        self.assertEqual(result['history'][-1]['elapsed_ms'],20)
        self.assertIn('solution',result)
        self.assertEqual(self.turn(game,kind='question',text='还可以问吗？').status_code,409)

    def test_true_clue_and_unverified_guess_do_not_win(self):
        game=self.start(story_id='blank')
        with patch('server.turtle.api.score',return_value=('A',[{'choice':'A','probability':1}],1)):
            result=self.turn(game,kind='question',text='学生是盲人吗？').json()
            self.assertEqual(result['history'][-1]['answer'],'是')
            self.assertEqual(result['status'],'playing');self.assertNotIn('solution',result)
            result=self.turn(game,kind='guess',text='他是盲人').json()
            self.assertEqual(result['status'],'playing');self.assertNotIn('solution',result)
            self.assertIn('还不能确认',result['history'][-1]['answer'])
            # The next check sees confirmed player contributions, not just the current text.
            self.assertEqual(self.completion.call_args.args[1][0]['text'],'学生是盲人吗？')

    def test_contradicted_question_never_triggers_completion(self):
        game=self.start(story_id='blank')
        self.completion.return_value=(True,10)
        with patch('server.turtle.api.score',return_value=('B',[{'choice':'B','probability':1}],1)):
            result=self.turn(game,kind='question',text='是用隐形墨水写的吗？').json()
        self.completion.assert_not_called()
        self.assertEqual(result['status'],'playing');self.assertNotIn('solution',result)

    def test_busy_and_failed_model_dont_mutate_history(self):
        game=self.start(story_id='blank');g=GAMES[game['id']]
        with g.lock:self.assertEqual(self.turn(game,kind='hint').status_code,409)
        with patch('server.turtle.api.score',side_effect=ValueError('failure')):
            with self.assertRaises(ValueError):self.turn(game,kind='question',text='问题？')
        self.assertEqual(g.history,[]);self.assertFalse(g.lock.locked())
    def test_limit_still_allows_reveal(self):
        game=self.start(story_id='blank');GAMES[game['id']].history=[{'kind':'hint'}]*80
        self.assertEqual(self.turn(game,kind='question',text='问题？').status_code,409)
        self.assertEqual(self.turn(game,kind='reveal').status_code,200)

if __name__=='__main__':unittest.main()
