import importlib.util
import random
import unittest

from agents.llm_agent import LLMAgent
from lm.rl import (
	Baseline, SessionSource, assign_advantages, compute_reward, log_softmax_list,
	play_one_game,
)

HAS_TORCH = (
	importlib.util.find_spec("torch") is not None
	and importlib.util.find_spec("transformers") is not None
)


class FakeScorer:
	def __init__(self, seed=0):
		self.rng = random.Random(seed)

	def score(self, prefix_lines, candidate_lines):
		return [self.rng.uniform(-5, 0) for _ in candidate_lines]


class RewardBaselineTests(unittest.TestCase):
	def test_reward(self):
		self.assertEqual(compute_reward(True, False, 1), 1.0)
		self.assertEqual(compute_reward(False, False, 1), 0.0)
		self.assertEqual(compute_reward(False, True, 4), 0.25)
		self.assertEqual(compute_reward(False, True, 0), 0.0)

	def test_baseline(self):
		b = Baseline(0.05)
		self.assertAlmostEqual(b.get(4), 0.25)
		self.assertAlmostEqual(b.advantage(4, 1.0), 0.75)
		self.assertAlmostEqual(b.get(4), 0.25 + 0.05 * 0.75)
		self.assertAlmostEqual(b.get(2), 0.5)  # separate per player count
		b2 = Baseline()
		b2.load_state_dict(b.state_dict())
		self.assertEqual(b2.value, b.value)

	def test_log_softmax(self):
		lp = log_softmax_list([0.0, 0.0, -1000.0])
		self.assertAlmostEqual(lp[0], -0.6931471805599453)


class RolloutTests(unittest.TestCase):
	def test_decisions_recorded(self):
		rng = random.Random(3)
		source = SessionSource(
			rng, lambda: LLMAgent("LLM", name="LLM", seed=1, scorer=FakeScorer(1)),
			games_per_session=3)
		seen_learners = set()
		results = []
		for _ in range(6):
			results.append(play_one_game(source, lambda p, c: log_softmax_list([0.0] * len(c))))
			seen_learners.add(id(source.learner))
		self.assertEqual(len(seen_learners), 2)  # new learner every 3 games
		for r in results:
			self.assertTrue(r.decisions)
			for d in r.decisions:
				self.assertGreaterEqual(len(d.candidates), 2)
				self.assertTrue(0 <= d.chosen < len(d.candidates))
				self.assertEqual(len(d.ref_logp), len(d.candidates))
				self.assertTrue(d.prefix[0].startswith("coup"))
			self.assertIn(r.ret, (0.0, 1.0) if not r.draw else (r.ret,))
		flat = assign_advantages(results, Baseline())
		self.assertEqual(len(flat), sum(len(r.decisions) for r in results))
		for r in results:
			advs = {d.advantage for d in r.decisions}
			self.assertEqual(len(advs), 1)

	def test_on_decision_default_none(self):
		self.assertIsNone(LLMAgent("a", scorer=FakeScorer()).on_decision)


@unittest.skipUnless(HAS_TORCH, "torch/transformers not installed")
class ScoreTensorTests(unittest.TestCase):
	def test_matches_score_and_has_grad(self):
		import torch
		from transformers import GPT2Config, GPT2LMHeadModel
		from lm.model import CoupLM

		class Tok:
			bos_token_id, eos_token_id = 0, 0

			def __call__(self, text, add_special_tokens=False):
				return {"input_ids": [1 + (ord(c) % 40) for c in text]}

		torch.manual_seed(0)
		model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=2,
			n_positions=256))
		lm = CoupLM(model, Tok(), "cpu", max_len=200)
		prefix = ["coup rules", "n 3 res win", "P0 inc"]
		cands = ["P0 coup P1", "P0 fa", "P0 steal P2 xx"]
		ref = lm.score(prefix, cands)
		t = lm.score_tensor(prefix, cands)
		self.assertTrue(t.requires_grad)
		self.assertEqual(t.shape, (3,))
		for a, b in zip(ref, t.tolist()):
			self.assertAlmostEqual(a, b, places=3)
		t.sum().backward()
		self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
			for p in model.parameters()))


if __name__ == "__main__":
	unittest.main()
