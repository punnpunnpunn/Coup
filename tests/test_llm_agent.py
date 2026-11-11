import importlib.util
import io
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agents.aggressive_belief import AggressiveBelief
from agents.always_duke import AlwaysDuke
from agents.final_agent import FinalAgent
from agents.honest_random import HonestRandom
from agents.llm_agent import LLMAgent
from agents.random_agent import RandomAgent
from src.game import Game

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = (
	importlib.util.find_spec("torch") is not None
	and importlib.util.find_spec("transformers") is not None
)


class FakeScorer:
	def __init__(self, seed=0, fn=None):
		self.rng = random.Random(seed)
		self.fn = fn
		self.calls = []

	def score(self, prefix_lines, candidate_lines):
		self.calls.append((list(prefix_lines), list(candidate_lines)))
		if self.fn:
			return [self.fn(c) for c in candidate_lines]
		return [self.rng.uniform(-5, 0) for _ in candidate_lines]


def play(game, cap=100):
	turns = {p.id: 0 for p in game.players}
	while game.winner is None:
		player = game.current
		player.choose_action(game)
		turns[player.id] += 1
		if game.winner is None and all(turns[p.id] >= cap for p in game.alive_players):
			return


class LLMAgentGameTests(unittest.TestCase):
	def test_full_games_with_fake_scorer(self):
		bots = [AlwaysDuke, AggressiveBelief, HonestRandom, RandomAgent, FinalAgent]
		rng = random.Random(1)
		for game_no in range(60):
			n = 2 + game_no % 5
			scorer = FakeScorer(seed=game_no)
			llm = LLMAgent("llm", name="llm", seed=game_no, scorer=scorer)
			players = [llm]
			for i in range(n - 1):
				cls = rng.choice(bots)
				players.append(cls(f"{cls.__name__} {i}"))
			rng.shuffle(players)
			with redirect_stdout(io.StringIO()):
				game = Game(players)
				play(game)
			self.assertTrue(game.winner is not None or len(game.alive_players) >= 2)

	def test_duplicate_names(self):
		for n in (2, 4):
			llm = LLMAgent("llm", name="Same", seed=1, scorer=FakeScorer())
			players = [llm] + [AlwaysDuke(f"id{i}", name="Same") for i in range(n - 1)]
			with redirect_stdout(io.StringIO()):
				game = Game(players)
				play(game)
			self.assertIsNotNone(game.winner)

	def test_history_grows_and_appears_in_prompt(self):
		scorer = FakeScorer()
		llm = LLMAgent("llm", name="llm", seed=3, scorer=scorer)
		bots = [AlwaysDuke("b1"), AlwaysDuke("b2")]
		past_counts = []
		for _ in range(3):
			scorer.calls.clear()
			with redirect_stdout(io.StringIO()):
				game = Game([llm] + bots)
				play(game)
			self.assertTrue(scorer.calls)
			prompt = scorer.calls[-1][0]
			past_counts.append(sum(1 for l in prompt if l.startswith("past ")))
			self.assertIn("now", prompt)
		self.assertEqual(past_counts[0], 0)
		self.assertEqual(past_counts[1:], [1, 2])

	def test_greedy_picks_max_score(self):
		prefs = {"P0 tax": 10.0}
		scorer = FakeScorer(fn=lambda c: prefs.get(c, -1.0))
		llm = LLMAgent("llm", name="llm", greedy=True, scorer=scorer)
		other = AlwaysDuke("other")
		game = Game([llm, other])
		with redirect_stdout(io.StringIO()):
			llm.choose_action(game)
		self.assertTrue(any("Duke for Tax" in m for m in game.log))
		self.assertIn("P0 tax", scorer.calls[0][1])

	def test_greedy_prefers_other_candidate(self):
		scorer = FakeScorer(fn=lambda c: 5.0 if c == "P0 inc" else 0.0)
		llm = LLMAgent("llm", name="llm", greedy=True, scorer=scorer)
		game = Game([llm, AlwaysDuke("other")])
		llm.choose_action(game)
		self.assertTrue(any("takes Income" in m for m in game.log))

	def test_import_does_not_import_torch(self):
		code = (
			"import sys, agents.llm_agent; "
			"sys.exit(1 if 'torch' in sys.modules or 'transformers' in sys.modules else 0)"
		)
		result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
		self.assertEqual(result.returncode, 0)

	def test_single_player_subclass(self):
		import agents.llm_agent as module
		from src.player import Player
		subclasses = [
			v for v in vars(module).values()
			if isinstance(v, type) and issubclass(v, Player) and v.__module__ == module.__name__
		]
		self.assertEqual(subclasses, [LLMAgent])


@unittest.skipUnless(HAS_TORCH, "torch/transformers not installed")
class CoupLMTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		from lm.model import CoupLM
		cls.lm = CoupLM.load()

	def test_encode_lines_spans(self):
		from lm.model import encode_lines
		ids, spans = encode_lines(self.lm.tokenizer, ["P0 inc", "P1 fa"])
		self.assertEqual(spans[0][0], 0)
		self.assertEqual(spans[0][1], spans[1][0])
		self.assertEqual(spans[1][1], len(ids))

	def test_score_matches_unbatched_full_pass(self):
		import torch
		import torch.nn.functional as F
		from lm.model import encode_lines
		prefix = ["n 3 res win", "deal duke con", "P0 inc"]
		cands = ["P1 tax", "P1 steal P0", "P1 inc"]
		scores = self.lm.score(prefix, cands)
		self.assertEqual(len(scores), 3)
		for cand, score in zip(cands, scores):
			ids, spans = encode_lines(self.lm.tokenizer, prefix + [cand])
			x = torch.tensor([ids], device=self.lm.device)
			with torch.no_grad():
				lp = F.log_softmax(self.lm.model(input_ids=x).logits[0].float(), -1)
			start = spans[-1][0]
			ref = sum(lp[t - 1, ids[t]].item() for t in range(start, len(ids)))
			self.assertAlmostEqual(score, ref, delta=0.5)  # bf16 noise


if __name__ == "__main__":
	unittest.main()
