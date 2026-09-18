import unittest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.snake import SnakeState, observe, select_action, router


class SnakeRulesTests(unittest.TestCase):
    def state(self, **kwargs):
        args = dict(size=6, snake=[[2, 2], [1, 2], [0, 2]], food=[4, 2], direction=1)
        args.update(kwargs)
        return SnakeState(**args)

    def test_observation_distances(self):
        observations = observe(self.state())
        self.assertEqual([o['direction'] for o in observations], [1, 0, 2])
        self.assertEqual([o['food_distance'] for o in observations], [1, 3, 3])

    def test_wall_and_guard(self):
        obs = observe(self.state(snake=[[5, 2], [4, 2], [3, 2]], food=[0, 0]))
        raw, selected, overridden = select_action(obs, {'A': .8, 'B': .15, 'C': .05}, True)
        self.assertEqual(raw['action'], 'straight')
        self.assertEqual(selected['action'], 'left')
        self.assertTrue(overridden)
        self.assertIsNone(obs[0]['food_distance'])
        _, selected, overridden = select_action(obs, {'A': .8, 'B': .15, 'C': .05}, False)
        self.assertFalse(overridden)
        self.assertEqual(selected['action'], 'straight')

    def test_tail_vacates(self):
        state = self.state(snake=[[2, 2], [2, 3], [1, 3], [1, 2]], food=[4, 4], direction=0)
        self.assertTrue(observe(state)[1]['safe'])

    def test_body_collision(self):
        state = self.state(snake=[[2, 2], [2, 3], [1, 3], [1, 2], [1, 1]], food=[4, 4], direction=0)
        self.assertFalse(observe(state)[1]['safe'])

    def test_trapped(self):
        state = self.state(snake=[[0, 0], [0, 1], [1, 1], [1, 0], [2, 0]], food=[5, 5], direction=0)
        _, selected, _ = select_action(observe(state), {'A': .3, 'B': .3, 'C': .4}, True)
        self.assertIsNone(selected)

    def test_reject_invalid_input(self):
        for params in [dict(food=[2, 2]), dict(snake=[[2, 2], [0, 2]]), dict(direction=0),
                       dict(food=[6, 2]), dict(size=True), dict(snake=[[2,2],[2,2]])]:
            with self.subTest(params=params), self.assertRaises(ValidationError):
                self.state(**params)

    def test_invalid_probabilities_rejected(self):
        for values in [{'A': 1, 'B': 1, 'C': 0}, {'A': float('nan'), 'B': 0, 'C': 0}]:
            with self.assertRaises(ValueError):
                select_action(observe(self.state()), values, True)

    def test_http_validation_without_loading_model(self):
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            response = client.post('/api/snake/decide', json={'size': 999, 'snake': [], 'food': [0,0], 'direction': 1})
            self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
