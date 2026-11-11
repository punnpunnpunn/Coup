import json
import math
import os
import tempfile
import unittest

from agents.llm_agent import LLMAgent
from lm import compare
from lm.generate_data import TEACHERS, make_player
from tests.test_llm_agent import FakeScorer


def bot_factory(cls):
	return lambda: make_player(cls, compare.CONTENDER_ID, compare.CONTENDER_ID, 0)


def llm_factory():
	return LLMAgent(compare.CONTENDER_ID, name="c", seed=0, scorer=FakeScorer(seed=5))


def run(factory, spec, slots):
	return compare.run_contender("x", factory, spec, slots, progress_every=0)


class ScheduleTests(unittest.TestCase):
	def test_schedule_deterministic_and_valid(self):
		a = compare.make_schedule(3, 20, [2, 3, 4, 5, 6], 6)
		b = compare.make_schedule(3, 20, [2, 3, 4, 5, 6], 6)
		self.assertEqual(a[0], b[0])
		self.assertEqual([s.__dict__ for s in a[1]], [s.__dict__ for s in b[1]])
		for s in a[1]:
			self.assertEqual(len(s.seats), s.n_players)
			self.assertEqual(s.seats.count(-1), 1)

	def test_identical_setup_across_contenders(self):
		spec, slots = compare.make_schedule(1, 12, [2, 3, 4], 5)
		runs = [run(bot_factory(TEACHERS[0]), spec, slots),
			run(bot_factory(TEACHERS[-1]), spec, slots),
			run(llm_factory, spec, slots)]
		for i, s in enumerate(slots):
			for r in runs:
				self.assertEqual(len(r[i]["hands"]), s.n_players)
			# the same game index deals the same cards to the same seats
			self.assertEqual(runs[0][i]["hands"], runs[1][i]["hands"])
			self.assertEqual(runs[0][i]["hands"], runs[2][i]["hands"])

	def test_identical_contenders_identical_results(self):
		spec, slots = compare.make_schedule(2, 25, [2, 3, 4, 5, 6], 6)
		a = run(bot_factory(TEACHERS[-1]), spec, slots)
		b = run(bot_factory(TEACHERS[-1]), spec, slots)
		self.assertEqual(a, b)
		c = run(llm_factory, spec, slots)
		d = run(llm_factory, spec, slots)
		self.assertEqual(c, d)


class StatsTests(unittest.TestCase):
	def test_wilson(self):
		lo, hi = compare.wilson(50, 100)
		self.assertAlmostEqual(lo, 0.4038, places=3)
		self.assertAlmostEqual(hi, 0.5962, places=3)
		lo, hi = compare.wilson(0, 10)
		self.assertEqual(lo, 0.0)
		self.assertAlmostEqual(hi, 0.2775, places=3)
		self.assertEqual(compare.wilson(0, 0), (0.0, 1.0))

	def test_sign_test(self):
		self.assertEqual(compare.sign_test_p(0, 0), 1.0)
		self.assertAlmostEqual(compare.sign_test_p(5, 5), 1.0)
		self.assertAlmostEqual(compare.sign_test_p(0, 10), 2 / 1024)
		self.assertAlmostEqual(compare.sign_test_p(10, 0), 2 / 1024)
		# n=10, k<=2: (1+10+45)/1024 doubled
		self.assertAlmostEqual(compare.sign_test_p(2, 8), 112 / 1024)
		self.assertTrue(math.isclose(compare.sign_test_p(1, 1), 1.0))


class JsonTests(unittest.TestCase):
	def test_json_structure(self):
		contenders = [("llm", llm_factory), ("Duke", bot_factory(TEACHERS[0])),
			("Final", bot_factory(TEACHERS[-1]))]
		result = compare.compare(contenders, 0, 15, [2, 3, 4], 5, progress_every=0)
		with tempfile.TemporaryDirectory() as d:
			path = os.path.join(d, "c.json")
			with open(path, "w") as f:
				json.dump(result, f)
			with open(path) as f:
				data = json.load(f)
		self.assertEqual(data["reference"], "llm")
		self.assertEqual(set(data["contenders"]), {"llm", "Duke", "Final"})
		self.assertEqual(set(data["paired"]), {"Duke", "Final"})
		self.assertEqual(len(data["schedule"]), 15)
		self.assertTrue(0 < data["chance_baseline"] < 0.5)
		for s in data["contenders"].values():
			for key in ("games", "wins", "draws", "win_rate", "ci95", "by_players",
					"by_opponent_bot", "outcomes"):
				self.assertIn(key, s)
			self.assertEqual(s["games"], 15)
			self.assertEqual(len(s["outcomes"]), 15)
		p = data["paired"]["Duke"]
		for key in ("reference_only_wins", "other_only_wins", "diff_win_rate", "p_value"):
			self.assertIn(key, p)


if __name__ == "__main__":
	unittest.main()
