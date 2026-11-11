"""Numerical equivalence of the incremental KV-cache `score` and packed `score_packed`
with the plain per-decision paths. Uses the real checkpoints/coup-lm on CPU (fp32)."""
import contextlib
import importlib.util
import io
import random
import unittest
from pathlib import Path

from agents.always_duke import AlwaysDuke
from agents.honest_random import HonestRandom
from agents.llm_agent import LLMAgent
from lm.evaluate import play
from src.game import Game

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "coup-lm"
HAS_TORCH = (
	importlib.util.find_spec("torch") is not None
	and importlib.util.find_spec("transformers") is not None
)


class Recorder:
	"""Fake scorer that records every (prefix, candidates) the agent asks about."""

	def __init__(self, seed=0):
		self.rng = random.Random(seed)
		self.calls = []

	def score(self, prefix_lines, candidate_lines):
		self.calls.append((list(prefix_lines), list(candidate_lines)))
		return [self.rng.uniform(-5, 0) for _ in candidate_lines]


def record_decisions(games=2, seed=5, min_decisions=12):
	"""Decisions (in order) of one persistent LLMAgent over several games (so history grows).

	Fully seeded (engine and every bot), and games with fewer than `min_decisions`
	scored decisions are skipped, so tests can index into them safely."""
	rec = Recorder(seed)
	llm = LLMAgent("llm", name="llm", seed=seed, scorer=rec)
	per_game = []
	g = 0
	while len(per_game) < games:
		if g > 200:
			raise RuntimeError("could not record long-enough games")
		start = len(rec.calls)
		random.seed(seed + g)
		with contextlib.redirect_stdout(io.StringIO()):
			game = Game([llm, AlwaysDuke("d", seed=seed + g), HonestRandom("h", seed=seed + g)])
			play(game)
		calls = rec.calls[start:]
		if len(calls) >= min_decisions:
			per_game.append(calls)
		g += 1
	return per_game


@unittest.skipUnless(HAS_TORCH and CKPT.exists(), "needs torch and checkpoints/coup-lm")
class KVCacheTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		from lm.model import CoupLM
		cls.lm = CoupLM.load(str(CKPT), device="cpu")
		cls.games = record_decisions()

	def test_session_matches_plain_score(self):
		lm = self.lm
		session = lm.new_session()
		calls = self.games[0][:12]
		self.assertGreater(len(calls), 3)
		for prefix, cands in calls:
			ref = lm.score(prefix, cands)
			got = lm.score(prefix, cands, session=session)
			for a, b in zip(ref, got):
				self.assertAlmostEqual(a, b, delta=1e-3)
		# the cache really was reused incrementally and stays batch-1
		self.assertEqual(session.cache.layers[0].keys.shape[0], 1)
		self.assertEqual(session.cache.get_seq_length(), len(session.ids))
		# same prefix again (no new tokens) and a forced head change (rebuild)
		prefix, cands = calls[-1]
		again = lm.score(prefix, cands, session=session)
		for a, b in zip(lm.score(prefix, cands), again):
			self.assertAlmostEqual(a, b, delta=1e-3)
		changed = ["# changed head"] + prefix
		for a, b in zip(lm.score(changed, cands), lm.score(changed, cands, session=session)):
			self.assertAlmostEqual(a, b, delta=1e-3)
		# back to the old prefix after the rebuild
		for a, b in zip(lm.score(prefix, cands), lm.score(prefix, cands, session=session)):
			self.assertAlmostEqual(a, b, delta=1e-3)

	def test_session_across_games_and_truncation(self):
		lm = self.lm
		old_max = lm.max_len
		try:
			session = lm.new_session()
			calls = self.games[0][-4:] + self.games[1][:6] + self.games[1][-6:]
			longest = max(len(lm._prefix_ids(p, 20)) for p, _ in calls)
			lm.max_len = max(longest - 150, 400)  # force truncation (changes the head)
			for prefix, cands in calls:
				for a, b in zip(lm.score(prefix, cands), lm.score(prefix, cands, session=session)):
					self.assertAlmostEqual(a, b, delta=1e-3)
		finally:
			lm.max_len = old_max

	def test_session_not_mutated_by_candidates(self):
		lm = self.lm
		session = lm.new_session()
		prefix, cands = self.games[0][5]
		lm.score(prefix, cands, session=session)
		n = session.cache.get_seq_length()
		self.assertEqual(n, len(session.ids))
		lm.score(prefix, cands, session=session)
		self.assertEqual(session.cache.get_seq_length(), n)

	def test_agent_sessions_are_per_agent(self):
		a = LLMAgent("a", scorer=self.lm)
		b = LLMAgent("b", scorer=self.lm)
		self.assertIsNone(a._session)
		a._session = a._get_scorer().new_session()
		self.assertIsNone(b._session)


@unittest.skipUnless(HAS_TORCH and CKPT.exists(), "needs torch and checkpoints/coup-lm")
class PackedTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		from lm.model import CoupLM
		cls.lm = CoupLM.load(str(CKPT), device="cpu")
		cls.games = record_decisions()

	def check(self, items):
		import torch
		lm = self.lm
		lm.model.train()
		try:
			packed = lm.score_packed(items)
			self.assertEqual(len(packed), len(items))
			for (prefix, cands), got in zip(items, packed):
				ref = lm.score_tensor(prefix, cands)
				self.assertEqual(ref.shape, got.shape)
				self.assertLess((ref - got).abs().max().item(), 1e-3)
			return packed
		finally:
			lm.model.eval()

	def test_packed_matches_per_decision(self):
		self.check(self.games[0][:10])

	def test_split_packs_and_fallback(self):
		lm = self.lm
		old = lm.max_len
		try:
			items = self.games[1][:6]
			longest = max(len(lm._prefix_ids(p, 20)) for p, _ in items)
			lm.max_len = longest + 60  # packs must split
			self.check(items)
			# decisions from different games: prefixes are not extensions of each other
			lm.max_len = old
			self.check([self.games[0][3], self.games[1][2], self.games[0][4]])
		finally:
			lm.max_len = old

	def test_gradients_flow_and_match(self):
		import torch
		lm = self.lm
		items = self.games[0][:4]
		lm.model.train()
		try:
			lm.model.zero_grad()
			loss = sum((s.log_softmax(0)[0]) for s in lm.score_packed(items))
			loss.backward()
			g_packed = [p.grad.clone() for p in lm.model.parameters() if p.grad is not None]
			self.assertTrue(any(g.abs().sum() > 0 for g in g_packed))
			lm.model.zero_grad()
			loss = sum((lm.score_tensor(p, c).log_softmax(0)[0]) for p, c in items)
			loss.backward()
			g_ref = [p.grad.clone() for p in lm.model.parameters() if p.grad is not None]
			num = sum(((a - b) ** 2).sum() for a, b in zip(g_packed, g_ref)).sqrt()
			den = sum((b ** 2).sum() for b in g_ref).sqrt()
			self.assertLess((num / den).item(), 1e-2)
		finally:
			lm.model.zero_grad()
			lm.model.eval()


if __name__ == "__main__":
	unittest.main()
