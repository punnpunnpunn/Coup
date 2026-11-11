import contextlib
import io
import tempfile
import unittest

from agents.llm_agent import LLMAgent
from lm import analyze, compare
from lm.generate_data import TEACHERS, make_player
from src.cards import Card, Role
from src.game import Game
from src.player import Player
from tests.test_llm_agent import FakeScorer


def bot_factory(name):
	cls = next(c for c in TEACHERS if c.__name__ == name)
	return lambda: make_player(cls, compare.CONTENDER_ID, compare.CONTENDER_ID, 0)


def llm_factory():
	return LLMAgent(compare.CONTENDER_ID, name="c", seed=0, scorer=FakeScorer(seed=5))


class Scripted(Player):
	"""Takes one fixed action, answers challenges as configured, never blocks."""

	def __init__(self, pid, action="income", challenge=False):
		super().__init__(pid, pid)
		self.action = action
		self.challenge = challenge

	def choose_action(self, game):
		return game.perform_action(self.id, self.action)

	def decide(self, kind, context):
		if kind == "challenge":
			return self.challenge
		if kind == "reveal_influence":
			return context["cards"][0]["index"]
		return None


def scenario(first, second, hands):
	"""Two scripted players with hand-built cards (id -> roles); `first` acts first."""
	with contextlib.redirect_stdout(io.StringIO()):
		game = Game([first, second])
	for p in (first, second):
		p.cards = [Card(Role(r)) for r in hands[p.id]]
	return game


class NoBehaviourChangeTests(unittest.TestCase):
	def run_logs(self, factory, instrument, seed=4, games=18):
		spec, slots = compare.make_schedule(seed, games, [2, 3, 4, 5, 6], 6)
		opponents = {i: make_player(c, pid, pid, s) for i, (c, pid, s) in enumerate(spec)}
		contender = factory()
		rec = analyze.Recorder(contender).install() if instrument else None
		captured = []
		orig = contender.on_game_start

		def grab(game):
			captured.append(game)
			orig(game)

		contender.on_game_start = grab
		outcomes = [compare.play_slot(s, contender, opponents, spec) for s in slots]
		logs = [(list(g.log), [dict(e) for e in g.events]) for g in captured]
		if rec:
			rec.flush()
		return outcomes, logs, rec

	def check(self, factory):
		a = self.run_logs(factory, False)
		b = self.run_logs(factory, True)
		self.assertEqual(a[0], b[0])
		self.assertEqual(a[1], b[1])
		self.assertEqual(len(b[2].games), len(a[1]))

	def test_bots_identical_logs(self):
		for name in ("FinalAgent", "RandomAgent", "AlwaysDuke", "DukeFish"):
			with self.subTest(name):
				self.check(bot_factory(name))

	def test_llm_path_identical_logs(self):
		self.check(llm_factory)

	def test_uninstall_restores(self):
		p = make_player(TEACHERS[0], "contender", "contender", 0)
		rec = analyze.Recorder(p).install()
		self.assertIn("decide", p.__dict__)
		rec.uninstall()
		self.assertNotIn("decide", p.__dict__)
		self.assertNotIn("choose_action", p.__dict__)


class BluffTests(unittest.TestCase):
	def test_always_tax_without_duke_is_all_bluff(self):
		a, b = Scripted("a", "tax"), Scripted("b", "income")
		rec = analyze.Recorder(a).install()
		game = scenario(a, b, {"a": ["captain", "contessa"], "b": ["duke", "duke"]})
		rec.begin(game)
		for _ in range(2):
			a.choose_action(game)
			b.choose_action(game)
		rec.flush()
		m = analyze.compute_metrics(rec.games)
		duke = m["bluffing"]["by_role"]["duke"]
		self.assertEqual((duke["claims"], duke["bluffs"], duke["bluff_rate"]), (2, 2, 1.0))
		self.assertEqual(m["bluffing"]["all"]["bluff_rate"], 1.0)
		self.assertEqual(m["actions"]["overall"]["counts"]["tax"], 2)

	def test_honest_claim_challenged_and_bluff_caught(self):
		a, b = Scripted("a", "tax"), Scripted("b", "income", challenge=True)
		rec = analyze.Recorder(a).install()
		game = scenario(a, b, {"a": ["duke", "contessa"], "b": ["captain", "captain"]})
		rec.begin(game)
		a.choose_action(game)  # honest tax, challenged by b -> proven
		rec.flush()
		c = analyze.compute_metrics(rec.games)["bluffing"]["by_role"]["duke"]
		self.assertEqual((c["claims"], c["bluffs"], c["honest_challenged"]), (1, 0, 1))

		a, b = Scripted("a", "tax"), Scripted("b", "income", challenge=True)
		rec = analyze.Recorder(a).install()
		game = scenario(a, b, {"a": ["captain", "contessa"], "b": ["captain", "captain"]})
		rec.begin(game)
		a.choose_action(game)  # bluff, challenged, caught
		rec.flush()
		c = analyze.compute_metrics(rec.games)["bluffing"]["by_role"]["duke"]
		self.assertEqual((c["bluffs"], c["bluffs_challenged"], c["bluffs_caught"]), (1, 1, 1))

	def test_only_contender_is_instrumented(self):
		a, b = Scripted("a", "income"), Scripted("b", "tax")
		rec = analyze.Recorder(a).install()
		game = scenario(a, b, {"a": ["captain", "captain"], "b": ["captain", "captain"]})
		rec.begin(game)
		a.choose_action(game)
		b.choose_action(game)
		rec.flush()
		self.assertEqual(rec.games[0]["claims"], [])


class ChallengeTests(unittest.TestCase):
	def run_case(self, b_hand, challenge):
		a, b = Scripted("a", "income", challenge=challenge), Scripted("b", "tax")
		rec = analyze.Recorder(a).install()
		game = scenario(b, a, {"a": ["captain", "contessa"], "b": b_hand})  # b acts first
		rec.begin(game)
		b.choose_action(game)
		rec.flush()
		return analyze.compute_metrics(rec.games)["challenging"]["all"]

	def test_correct_challenge_of_bluff(self):
		c = self.run_case(["captain", "contessa"], True)
		self.assertEqual((c["opportunities"], c["challenges"], c["accuracy"]), (1, 1, 1.0))
		self.assertEqual(c["missed_bluffs"], 0)

	def test_wrong_challenge_of_honest_claim(self):
		c = self.run_case(["duke", "contessa"], True)
		self.assertEqual((c["challenges"], c["accuracy"], c["claimant_bluffing"]), (1, 0.0, 0))

	def test_missed_bluff(self):
		c = self.run_case(["captain", "contessa"], False)
		self.assertEqual((c["challenges"], c["challenge_rate"]), (0, 0.0))
		self.assertEqual((c["claimant_bluffing"], c["missed_bluffs"], c["missed_bluff_rate"]), (1, 1, 1.0))
		self.assertIsNone(c["accuracy"])

	def test_claimant_hand_ignores_revealed_cards(self):
		a, b = Scripted("a", "income", challenge=True), Scripted("b", "tax")
		rec = analyze.Recorder(a).install()
		game = scenario(b, a, {"a": ["captain", "contessa"], "b": ["duke", "captain"]})
		b.cards[0].revealed = True  # the only duke is face up: the claim is a bluff
		rec.begin(game)
		b.choose_action(game)
		rec.flush()
		c = analyze.compute_metrics(rec.games)["challenging"]["all"]
		self.assertEqual(c["accuracy"], 1.0)


class ConsistencyTests(unittest.TestCase):
	def run_analyze(self, factory, name="x", games=40, examples_dir=None):
		return analyze.analyze([(name, factory)], 1, games, [2, 3, 4, 5, 6], 6,
			examples_dir=examples_dir, progress_every=0)

	def test_action_counts_equal_turns_and_winrate_matches_compare(self):
		for factory in (bot_factory("FinalAgent"), bot_factory("RandomAgent"), llm_factory):
			res = self.run_analyze(factory)["contenders"]["x"]
			cmp = compare.compare([("x", factory)], 1, 40, [2, 3, 4, 5, 6], 6, progress_every=0)
			self.assertEqual(res["outcomes"]["win_rate"], cmp["contenders"]["x"]["win_rate"])
			self.assertEqual(res["outcomes"]["wins"], cmp["contenders"]["x"]["wins"])
			counts = res["actions"]["overall"]
			self.assertEqual(counts["n"], sum(counts["counts"].values()))
			turns = res["outcomes"]["avg_turns_taken"] * res["outcomes"]["games"]
			self.assertAlmostEqual(counts["n"], turns)
			for split in ("by_coin_bucket", "by_phase"):
				self.assertEqual(sum(d["n"] for d in res["actions"][split].values()), counts["n"])

	def test_turns_match_event_stream(self):
		spec, slots = compare.make_schedule(2, 15, [2, 3, 4], 6)
		holder = []

		def f():
			p = bot_factory("HonestRandom")()
			holder.append(analyze.Recorder(p).install())
			return p

		compare.run_contender("x", f, spec, slots, progress_every=0)
		rec = holder[0]
		rec.flush()
		for g in rec.games:
			acts = [d for d in g["decisions"] if d["kind"] == "action"]
			mine = [e for e in g["events"] if e["type"] == "action" and e["actor"] == "contender"]
			self.assertEqual(len(acts), len(mine))
			self.assertEqual(len(acts), g["own_turns"])

	def test_examples_and_summary(self):
		with tempfile.TemporaryDirectory() as d:
			res = self.run_analyze(bot_factory("RandomAgent"), "rand", games=30, examples_dir=d)
			files = res["contenders"]["rand"]["examples"]
			self.assertEqual(len(files), 3)
			self.assertEqual(len(set(files)), 3)
			for f in files:
				with open(f, encoding="utf-8") as fh:
					text = fh.read()
				self.assertIn(">>", text)
				self.assertIn("Initial hands", text)
			buf = io.StringIO()
			with contextlib.redirect_stdout(buf):
				analyze.print_summary(res)
			self.assertIn("rand", buf.getvalue())


if __name__ == "__main__":
	unittest.main()
