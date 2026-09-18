import unittest
from pydantic import ValidationError
from server.snake import SnakeState, observe
from server.snake_planning import step, enrich, food_forecast, survival_depth, planning_candidates


class ForecastTests(unittest.TestCase):
    def state(self, **kwargs):
        args = dict(size=6, snake=[[2,2],[1,2],[0,2]], food=[4,2], direction=1)
        args.update(kwargs)
        return SnakeState(**args)

    def test_tail_moves_but_growth_retains_it(self):
        body = ((2,2),(2,3),(1,3),(1,2))
        moved = step(body, 3, (4,4), 6)
        self.assertIsNotNone(moved)
        self.assertEqual(moved[0], ((1,2),(2,2),(2,3),(1,3)))
        self.assertIsNone(step(body,3,(1,2),6))
        grew, eats = step(body,0,(2,1),6)
        self.assertTrue(eats)
        self.assertEqual(len(grew),5)
        self.assertEqual(grew[-1],body[-1])

    def test_food_route_and_history(self):
        state = self.state(recent_heads=[[3,2],[3,2],[2,1]], steps_since_food=10)
        observations = enrich(state, observe(state))
        straight = observations[0]
        self.assertEqual(straight['route_steps'],2)
        self.assertEqual(straight['recent_visits'],2)
        self.assertTrue(straight['escape_after_food'])
        self.assertEqual(straight['survival_steps'],6)

    def test_detect_delayed_trap_after_eating(self):
        state = self.state(snake=[[1,1],[1,2],[1,3],[2,3],[3,3],[3,2],
                                 [3,1],[3,0],[2,0],[1,0],[0,0],[0,1]],
                           food=[2,1], direction=0)
        observations = enrich(state, observe(state))
        right = observations[2]
        self.assertTrue(right['safe'])
        self.assertTrue(right['eats'])
        self.assertLess(right['survival_steps'],6)
        self.assertFalse(right['escape_after_food'])
        allowed, _ = planning_candidates(observations,0,6)
        self.assertNotIn(right,allowed)
        self.assertTrue(all(o['safe'] for o in allowed))

    def test_dynamic_search_can_use_departing_tail(self):
        body=((2,2),(2,3),(1,3),(1,2))
        count,escape=food_forecast(body,0,(0,2),6)
        self.assertEqual(count,2)
        self.assertTrue(escape)

    def test_planner_prefers_progress_and_breaks_loops(self):
        state = self.state()
        observations = enrich(state,observe(state))
        allowed, reason=planning_candidates(observations,0,6)
        self.assertEqual([o['action'] for o in allowed],['straight'])
        self.assertEqual(reason,'safe_food_route')
        for i,o in enumerate(observations):
            o['route_steps']=None
            o['recent_visits']=2-i
        allowed,reason=planning_candidates(observations,20,6)
        self.assertEqual(reason,'break_loop')
        self.assertEqual(allowed[0]['action'],'right')

    def test_unknown_routes_are_not_immediate_death(self):
        state=self.state()
        observations=enrich(state,observe(state))
        for o in observations:
            o['route_steps']=None
            o['escape_after_food']=None
        allowed,reason=planning_candidates(observations,0,6)
        self.assertEqual(len(allowed),3)
        self.assertEqual(reason,'survival_forecast')

    def test_memory_input_is_bounded(self):
        for args in [dict(recent_heads=[[0,0]]*25),dict(recent_heads=[[6,0]]),
                     dict(recent_heads=[[0]]),dict(steps_since_food=-1),dict(policy='unknown')]:
            with self.subTest(args=args), self.assertRaises(ValidationError):
                self.state(**args)


if __name__=='__main__':
    unittest.main()
