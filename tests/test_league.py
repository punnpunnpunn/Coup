import collections
import importlib.util
import json
import os
import random
import tempfile
import unittest

from agents.llm_agent import LLMAgent
from lm.league import League, LeagueSessionSource, pfsp_weight, pick_eviction
from lm.rl import Baseline, assign_advantages, log_softmax_list, play_one_game

HAS_TORCH = (
	importlib.util.find_spec("torch") is not None
	and importlib.util.find_spec("transformers") is not None
)
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "coup-lm")


class CountingScorer:
	def __init__(self, seed=0):
		self.rng = random.Random(seed)
		self.calls = 0

	def score(self, prefix_lines, candidate_lines, session=None):
		self.calls += 1
		return [self.rng.uniform(-5, 0) for _ in candidate_lines]


def make_league(max_snapshots=5, **kw):
	league = League(max_snapshots, **kw)
	league.add_bc(CountingScorer(1))
	return league


def ref_fn(p, c):
	return log_softmax_list([0.0] * len(c))


class EvictionTests(unittest.TestCase):
	def test_pick(self):
		self.assertIsNone(pick_eviction([10]))
		self.assertEqual(pick_eviction([10, 20]), 0)
		self.assertEqual(pick_eviction([10, 20, 30]), 1)
		self.assertEqual(pick_eviction([10, 20, 30, 40, 100]), 1)  # tie (20/20) -> older
		self.assertEqual(pick_eviction([10, 20, 70, 100]), 1)  # gaps 60 vs 80
		self.assertEqual(pick_eviction([10, 40, 50, 100]), 1)  # gaps 40 vs 60

	def test_spread_kept(self):
		league = make_league(5)
		for u in range(10, 310, 10):
			league.add_snapshot(u, CountingScorer(), None)
			ups = [s.update for s in league.snapshots]
			self.assertLessEqual(len(ups), 5)
			self.assertEqual(ups[0], 10)  # oldest kept
			self.assertEqual(ups[-1], u)  # newest kept
			self.assertIsNotNone(league.bc)  # BC never evicted
		ups = [s.update for s in league.snapshots]
		self.assertEqual(len(ups), 5)
		self.assertLess(ups[1], 150)  # spread over history, not only the recent ones
		self.assertLess(ups[2], 250)
		self.assertEqual(len(league.evicted), 30 - 5)  # evicted ones are recorded
		for u in range(10, 310, 10):
			if u not in ups:
				self.assertIn(f"u{u:04d}", [e["id"] for e in league.evicted])

	def test_max_one(self):
		league = make_league(1)
		for u in (10, 20, 30):
			league.add_snapshot(u, CountingScorer(), None)
		self.assertEqual([s.update for s in league.snapshots], [30])


class PfspTests(unittest.TestCase):
	def test_weights(self):
		self.assertAlmostEqual(pfsp_weight(0.3, 2), 0.49 + 0.05)
		self.assertAlmostEqual(pfsp_weight(1.0, 2), 0.05)  # easy opponents still appear
		self.assertAlmostEqual(pfsp_weight(0.0, 2), 1.05)

	def test_ema_and_sampling(self):
		league = make_league(decay=0.1)
		league.add_snapshot(10, CountingScorer(), None)
		bc, snap = league.members
		self.assertEqual((bc.winrate, snap.winrate), (0.3, 0.3))
		league.record_game({"types": {"bc"}, "members": [bc, bc]}, True)  # once per member per game
		self.assertAlmostEqual(bc.winrate, 0.3 + 0.1 * 0.7)
		self.assertEqual((bc.games, snap.games), (1, 0))
		league.record_game({"types": {"bc"}, "members": [bc]}, False)
		self.assertAlmostEqual(bc.winrate, 0.37 * 0.9)
		bc.winrate, snap.winrate = 0.9, 0.0  # the snapshot is hard -> sampled far more
		counts = collections.Counter(league.sample(random.Random(i)).id for i in range(4000))
		expect = pfsp_weight(0.0, 2) / (pfsp_weight(0.0, 2) + pfsp_weight(0.9, 2))
		self.assertAlmostEqual(counts["u0010"] / 4000, expect, delta=0.03)
		self.assertGreater(counts["bc"], 0)

	def test_type_stats(self):
		league = make_league()
		league.add_snapshot(10, CountingScorer(), None)
		league.record_game({"types": set(), "members": []}, True)
		league.record_game({"types": {"bc", "self"}, "members": [league.bc]}, False)
		self.assertEqual(league.type_stats["bots_only"], [1, 1])
		self.assertEqual(league.type_stats["with_bc"], [0, 1])
		self.assertEqual(league.type_stats["with_self"], [0, 1])
		self.assertEqual(league.type_stats["with_snap"], [0, 0])


def source_for(league, p_bots=0.4, p_self=0.2, seed=0, self_scorer=None, gps=3, **kw):
	learner_scorer = CountingScorer(7)
	self_scorer = self_scorer or CountingScorer(8)
	src = LeagueSessionSource(
		random.Random(seed), lambda: LLMAgent("LLM", name="LLM", seed=1, scorer=learner_scorer),
		league, self_scorer, p_bots=p_bots, p_self=p_self, games_per_session=gps, **kw)
	return src, learner_scorer, self_scorer


class SamplingTests(unittest.TestCase):
	def test_identity_proportions(self):
		league = make_league()
		league.add_snapshot(10, CountingScorer(), None)
		src, _, _ = source_for(league, seed=5, min_pool=7, max_pool=7)
		kinds = collections.Counter()
		for _ in range(600):
			src._new_session()
			for kind, *_ in src.pool:
				kinds[kind] += 1
		total = sum(kinds.values())
		llm = total - kinds["bot"]
		self.assertAlmostEqual(kinds["bot"] / total, 0.4, delta=0.03)
		self.assertAlmostEqual(kinds["self"] / llm, 0.2, delta=0.03)
		self.assertGreater(kinds["bc"], 0)
		self.assertGreater(kinds["snap"], 0)

	def test_stable_ids_within_session(self):
		league = make_league()
		src, _, _ = source_for(league, p_bots=0.0, p_self=0.5, gps=4)
		src._new_session()
		agents = [a for _, a, _, _ in src.pool]
		self.assertEqual(len({a.id for a in agents}), len(agents))
		before = [id(a) for _, a, _, _ in src.pool]
		for _ in range(4):
			src.next_game()
		self.assertEqual(before, [id(a) for _, a, _, _ in src.pool])  # same agents all session

	def test_all_bots(self):
		league = make_league()
		src, _, _ = source_for(league, p_bots=1.0)
		src.next_game()
		self.assertEqual(src.game_info["types"], set())


class RolloutTests(unittest.TestCase):
	def test_only_learner_decisions_recorded(self):
		league = make_league()
		league.add_snapshot(10, CountingScorer(3), None)
		src, learner_sc, self_sc = source_for(league, p_bots=0.0, p_self=0.3, seed=2)
		bc_sc, snap_sc = league.bc.scorer, league.snapshots[0].scorer
		infos = []
		for _ in range(12):
			learner_sc.calls = 0
			r = play_one_game(src, ref_fn)
			infos.append(r.info)
			# the learner scorer is called once per decision with >=2 candidates; those are
			# exactly the recorded decisions (opponent decisions never are)
			self.assertEqual(len(r.decisions), learner_sc.calls)
			self.assertTrue(r.decisions)
			for a in src._llms:
				self.assertIsNone(a.on_decision)
		self.assertGreater(bc_sc.calls + snap_sc.calls + self_sc.calls, 0)
		self.assertTrue({"bc", "snap", "self"} <= set().union(*(i["types"] for i in infos)))

	def test_sessions_are_per_agent_and_reset(self):
		class Sess:
			resets = 0

			def reset(self):
				self.resets += 1

		class SessScorer(CountingScorer):
			def new_session(self):
				return Sess()

			def score(self, prefix_lines, candidate_lines, session=None):
				assert session is not None
				return super().score(prefix_lines, candidate_lines)

		league = make_league()
		ss = SessScorer(4)
		src, _, _ = source_for(league, p_bots=0.0, p_self=1.0, self_scorer=ss, seed=4,
			min_pool=3, max_pool=3)
		for _ in range(3):
			play_one_game(src, ref_fn)
		sessions = [a._session for _, a, _, _ in src.pool if a._session is not None]
		self.assertTrue(sessions)
		self.assertEqual(len({id(s) for s in sessions}), len(sessions))  # never shared
		self.assertTrue(all(s.resets >= 1 for s in sessions))
		# "self" opponents hold the live scorer object -> they see weight updates
		self.assertTrue(all(a._scorer is ss for _, a, _, _ in src.pool))


class PersistenceTests(unittest.TestCase):
	def test_roundtrip(self):
		with tempfile.TemporaryDirectory() as d:
			league = make_league(3, decay=0.2, power=1.5)
			for u in (10, 20, 30, 40, 50):
				league.add_snapshot(u, f"lm{u}", os.path.join(d, f"u{u:04d}"))
			league.bc.winrate, league.bc.games = 0.5, 12
			league.snapshots[1].winrate = 0.1
			league.type_stats["with_bc"] = [3, 9]
			path = os.path.join(d, "league.json")
			league.save_json(path, {"update": 50})
			with open(path) as f:
				saved = json.load(f)
			self.assertEqual(saved["update"], 50)
			self.assertEqual(set(saved["pfsp_weights"]), {m.id for m in league.members})

			loaded = []
			fresh = League(3, decay=0.2, power=1.5)
			fresh.add_bc(CountingScorer())
			fresh.load_state_dict(saved, lambda p, u: loaded.append((p, u)) or f"lm{u}")
			self.assertEqual([m.id for m in fresh.members], [m.id for m in league.members])
			self.assertEqual((fresh.bc.winrate, fresh.bc.games), (0.5, 12))
			for a, b in zip(fresh.snapshots, league.snapshots):
				self.assertEqual((a.update, a.winrate, a.games, a.path), (b.update, b.winrate, b.games, b.path))
			self.assertEqual(fresh.type_stats, league.type_stats)
			self.assertEqual(fresh.evicted, league.evicted)
			self.assertEqual(fresh.weights(), league.weights())
			self.assertEqual([u for _, u in loaded], [s.update for s in league.snapshots])

	def test_missing_snapshot_dir_is_skipped(self):
		league = make_league()
		league.add_snapshot(10, "x", "nowhere")
		fresh = League()
		fresh.add_bc(CountingScorer())

		def loader(path, upd):
			raise FileNotFoundError(path)

		fresh.load_state_dict(league.state_dict(), loader)
		self.assertEqual([m.id for m in fresh.members], ["bc"])
		self.assertEqual(len(fresh.evicted), 1)


@unittest.skipUnless(HAS_TORCH, "torch/transformers not installed")
class TorchTests(unittest.TestCase):
	def test_tiny_end_to_end(self):
		import torch
		from transformers import GPT2Config, GPT2LMHeadModel
		from lm.league import snapshot_lm
		from lm.model import CoupLM
		from lm.rl import AutocastScorer, policy_update

		class Tok:
			bos_token_id, eos_token_id = 0, 0

			def __call__(self, text, add_special_tokens=False):
				return {"input_ids": [1 + (ord(c) % 40) for c in text]}

		torch.manual_seed(0)
		cfg = GPT2Config(vocab_size=64, n_embd=16, n_layer=1, n_head=2, n_positions=4096)
		policy = CoupLM(GPT2LMHeadModel(cfg).float(), Tok(), "cpu", max_len=3072)
		ref = CoupLM(GPT2LMHeadModel(cfg).float(), Tok(), "cpu", max_len=3072)
		opt = torch.optim.AdamW(policy.model.parameters(), lr=1e-2)
		league = League(2, make_scorer=lambda lm: AutocastScorer(lm, torch.bfloat16))
		league.add_bc(ref)
		pscorer = AutocastScorer(policy, torch.bfloat16)
		rng = random.Random(0)
		src = LeagueSessionSource(
			rng, lambda: LLMAgent("LLM", name="LLM", seed=rng.randrange(99), scorer=pscorer,
				history_max_lines=30),
			league, pscorer, games_per_session=2, min_pool=3, max_pool=3, p_bots=0.0, p_self=0.4,
			history_max_lines=30, max_players=3)
		args = type("A", (), {"micro_batch": 4, "pack": True, "pack_decisions": 16, "beta": 0.05,
			"grad_clip": 1.0})()
		baseline = Baseline()
		for update in range(1, 7):
			policy.model.eval()
			with torch.no_grad():
				results = [play_one_game(src, lambda p, c: log_softmax_list(ref.score(p, c)))
					for _ in range(2)]
			for r in results:
				league.record_game(r.info, r.won)
			decisions = assign_advantages(results, baseline)
			self.assertTrue(decisions)
			policy_update(policy, decisions, opt, args, None, torch.bfloat16)
			if update % 2 == 0:
				league.add_snapshot(update, snapshot_lm(policy), None)
		self.assertEqual([s.update for s in league.snapshots], [2, 6])  # spread: 2 kept, 4 evicted
		self.assertTrue(any(m.games for m in league.members))
		self.assertTrue(all(not p.requires_grad for m in league.snapshots
			for p in m.lm.model.parameters()))

	@unittest.skipUnless(os.path.isdir(CKPT), "checkpoints/coup-lm missing")
	def test_snapshot_copy_real_checkpoint(self):
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer
		from lm.league import snapshot_lm
		from lm.model import CoupLM

		tok = AutoTokenizer.from_pretrained(CKPT)
		model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32)
		policy = CoupLM(model, tok, "cpu")
		policy.model.train()
		snap = snapshot_lm(policy)
		self.assertFalse(snap.model.training)
		self.assertTrue(all(p.dtype == torch.bfloat16 and not p.requires_grad
			for p in snap.model.parameters()))
		ps = policy.model.state_dict()
		ss = snap.model.state_dict()
		self.assertEqual(set(ps), set(ss))
		for k in ps:  # equal to the source at copy time, no shared storage
			self.assertTrue(torch.equal(ss[k], ps[k].to(torch.bfloat16)), k)
			self.assertNotEqual(ss[k].data_ptr(), ps[k].data_ptr())
		before = {k: v.clone() for k, v in ss.items()}
		opt = torch.optim.SGD(policy.model.parameters(), lr=0.5)
		ids = torch.tensor([tok("P0 coup P1\n", add_special_tokens=False)["input_ids"]])
		policy.model(input_ids=ids, labels=ids).loss.backward()
		opt.step()
		changed = sum(not torch.equal(policy.model.state_dict()[k].to(torch.bfloat16), before[k])
			for k in before)
		self.assertGreater(changed, 0)  # the source did move...
		for k, v in snap.model.state_dict().items():
			self.assertTrue(torch.equal(v, before[k]), k)  # ...the snapshot did not


if __name__ == "__main__":
	unittest.main()
