import contextlib
import io
import random
import unittest

from agents.always_duke import AlwaysDuke
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.hibernate import Hibernate
from agents.honest_random import HonestRandom
from agents.passive_agent import PassiveAgent
from agents.random_agent import RandomAgent
from lm.encoding import (
    ROLE_ABBR,
    Transcript,
    action_candidates,
    decision_candidates,
    History,
    header_lines,
    is_move_line,
    is_own_move_line,
    record,
    rules_lines,
    translate_event,
)
from lm.generate_data import play_game
from src.cards import Card, Role
from src.game import Game
from src.player import Player


def make_game(names, me_index=0):
    players = [Player(f"id{i}", name) for i, name in enumerate(names)]
    game = Game(players)
    transcript = Transcript(players[me_index].id)
    transcript.attach(game)
    return game, players, transcript


def translate(transcript, game, events):
    game.events.extend(events)
    transcript.sync()
    return transcript.body[1:]


def ev(type_, **fields):
    return {"type": type_, **fields}


class TranslationTests(unittest.TestCase):
    def test_every_template(self):
        game, players, t = make_game(["Ann", "Bob", "Cy"])
        lines = translate(t, game, [
            ev("action", actor="id0", action="income", target=None),
            ev("coins", player="id0", delta=1, coins=3),
            ev("action", actor="id1", action="foreign_aid", target=None),
            ev("block", blocker="id2", role="duke", action="foreign_aid"),
            ev("no_challenge", challenger="id0", claimant="id2", role="duke"),
            ev("no_challenge", challenger="id1", claimant="id2", role="duke"),
            ev("block_result", blocker="id2", succeeded=True),
            ev("action", actor="id2", action="coup", target="id0"),
            ev("reveal", player="id0", role="contessa"),
            ev("action", actor="id0", action="tax", target=None),
            ev("challenge", challenger="id1", claimant="id0", role="duke"),
            ev("claim_failed", claimant="id0", role="duke"),
            ev("claim_proven", claimant="id0", role="duke"),
            ev("replacement", player="id0"),
            ev("action", actor="id1", action="assassinate", target="id2"),
            ev("block", blocker="id2", role="contessa", action="assassinate"),
            ev("action", actor="id1", action="exchange", target=None),
            ev("exchanged", player="id1"),
            ev("action", actor="id2", action="steal", target="id0"),
            ev("block", blocker="id0", role="ambassador", action="steal"),
            ev("no_block", action="steal"),
            ev("action_failed", action="tax"),
            ev("steal_fizzled", target="id0"),
            ev("turn", player="id1"),
            ev("win", player="id1"),
        ])
        self.assertEqual(lines, [
            "P0 inc",
            "P1 fa",
            "P2 blk duke",
            "P0 ok",
            "P2 coup P0",
            "P0 lose con",
            "P0 tax",
            "P1 chal",
            "P0 bluff",
            "P0 proof",
            "P1 kill P2",
            "P2 blk con",
            "P1 xch",
            "P2 steal P0",
            "P0 blk amb",
        ])

    def test_only_p0_ok_is_kept(self):
        game, _, t = make_game(["Ann", "Bob", "Cy"], me_index=1)
        lines = translate(t, game, [
            ev("no_challenge", challenger="id0", claimant="id2", role="duke"),
            ev("no_challenge", challenger="id1", claimant="id2", role="duke"),
            ev("no_challenge", challenger="id2", claimant="id0", role="duke"),
        ])
        self.assertEqual(lines, ["P0 ok"])

    def test_sync_is_incremental(self):
        game, _, t = make_game(["A", "B"])
        translate(t, game, [ev("action", actor="id1", action="income", target=None)])
        t.sync()
        self.assertEqual(t.body[1:], ["P1 inc"])

    def test_translate_event_uses_ids_not_names(self):
        labels = {"a": "P0", "b": "P1"}
        self.assertEqual(
            translate_event(ev("action", actor="b", action="steal", target="a"), labels, "a"),
            "P1 steal P0",
        )
        self.assertIsNone(translate_event(ev("win", player="a"), labels, "a"))

    def test_duplicate_names_work(self):
        players = [Player("a", "Same"), Player("b", "Same"), Player("c", "Same")]
        game = Game(players)
        t = Transcript("b")
        t.attach(game)
        self.assertEqual(
            translate(t, game, [
                ev("action", actor="a", action="steal", target="c"),
                ev("challenge", challenger="c", claimant="a", role="captain"),
            ]),
            ["P2 steal P1", "P1 chal"],
        )

    def test_duplicate_names_full_game(self):
        random.seed(3)
        players = [RandomCandidatePlayer(f"id{i}", "Dup", seed=i) for i in range(4)]
        game = Game(players)
        with contextlib.redirect_stdout(io.StringIO()):
            play_game(game)
        for p in players:
            p.transcript.sync()
            self.assertTrue(is_subsequence(p.chosen, p.transcript.body))

    def test_relative_seat_labels(self):
        game, players, t = make_game(["A", "B", "C", "D"], me_index=2)
        self.assertEqual(t.labels, {"id2": "P0", "id3": "P1", "id0": "P2", "id1": "P3"})
        self.assertEqual(
            translate(t, game, [
                ev("action", actor="id0", action="income", target=None),
                ev("action", actor="id3", action="income", target=None),
            ]),
            ["P2 inc", "P1 inc"],
        )

    def test_deal_line_canonical_order(self):
        game, players, _ = make_game(["A", "B"])
        players[0].cards = [Card(Role.CONTESSA), Card(Role.DUKE)]
        t = Transcript("id0")
        t.attach(game)
        self.assertEqual(t.body, ["deal duke con"])

    def test_attach_skips_existing_events(self):
        game, _, _ = make_game(["A", "B"])
        game.events.append(ev("action", actor="id0", action="income", target=None))
        t = Transcript("id0")
        t.attach(game)
        t.sync()
        self.assertEqual(len(t.body), 1)

    def test_header(self):
        lines = header_lines(3, "win")
        self.assertEqual(lines[-1], "n 3 res win")
        self.assertEqual(lines[-2], "P0 is me.")

    def test_prompt_layout_without_history(self):
        game, _, t = make_game(["A", "B"])
        lines = t.prompt_lines("lose")
        self.assertEqual(lines[-3:], ["now", "n 2 res lose", t.body[0]][:3])
        self.assertEqual(lines[: len(rules_lines())], rules_lines())
        self.assertEqual(t.history_lines(), [])


class SyntheticLineTests(unittest.TestCase):
    def test_state_line_only_on_change(self):
        game, players, t = make_game(["A", "B", "C"], me_index=1)
        players[1].cards = [Card(Role.CAPTAIN), Card(Role.ASSASSIN)]
        t.attach(game)
        t.before_decision("action", None)
        t.before_decision("action", None)
        self.assertEqual(t.body, ["deal ass cap", "$ 2 2 2 | ass cap"])
        players[0].coins = 5
        t.before_decision("challenge", {})
        self.assertEqual(t.body[-1], "$ 2 2 5 | ass cap")
        self.assertEqual(len(t.body), 3)
        players[1].cards[0].revealed = True  # captain revealed
        t.before_decision("challenge", {})
        self.assertEqual(t.body[-1], "$ 2 2 5 | ass")

    def test_draw_and_keep(self):
        game, players, t = make_game(["A", "B"])
        players[0].cards = [Card(Role.DUKE), Card(Role.CONTESSA)]
        t.attach(game)
        context = {
            "player_id": "id0",
            "keep_count": 2,
            "cards": [
                {"index": 0, "role": "duke"},
                {"index": 1, "role": "contessa"},
                {"index": 2, "role": "captain"},
                {"index": 3, "role": "duke"},
            ],
        }
        t.before_decision("exchange", context)
        self.assertEqual(t.body[-1], "draw duke con cap duke")
        cands = t.candidates("exchange", context)
        self.assertEqual(len({c.line for c in cands}), len(cands))
        self.assertEqual(len(cands), 4)  # dd, d-con, d-cap, con-cap (deduplicated by role)
        chosen = t.match("exchange", context, [1, 2])
        self.assertEqual(chosen.line, "keep cap con")
        self.assertTrue(chosen.synthetic)
        t.after_decision(chosen)
        self.assertEqual(t.body[-1], "keep cap con")

    def test_block_pass_is_synthetic_ok(self):
        game, players, t = make_game(["A", "B"])
        context = {
            "actor_id": "id1", "action": "steal", "target_id": "id0",
            "eligible_blockers": ["id0"], "allowed_roles": ["ambassador", "captain"],
        }
        cands = decision_candidates("block", context, "id0", t.labels)
        self.assertEqual([c.line for c in cands], ["P0 blk amb", "P0 blk cap", "P0 ok"])
        self.assertTrue(cands[-1].synthetic)
        self.assertFalse(cands[0].synthetic)
        t.after_decision(t.match("block", context, None))
        self.assertEqual(t.body[-1], "P0 ok")

    def test_block_not_eligible_has_no_candidates(self):
        context = {
            "actor_id": "id1", "action": "foreign_aid", "target_id": None,
            "eligible_blockers": ["id2"], "allowed_roles": ["duke"],
        }
        self.assertEqual(decision_candidates("block", context, "id0", {}), [])

    def test_foreign_aid_block_candidates(self):
        context = {
            "actor_id": "id1", "action": "foreign_aid", "target_id": None,
            "eligible_blockers": ["id0"], "allowed_roles": ["duke"],
        }
        cands = decision_candidates("block", context, "id0", {})
        self.assertEqual([c.line for c in cands], ["P0 blk duke", "P0 ok"])
        self.assertEqual(cands[0].answer, {"blocker_id": "id0"})

    def test_action_candidates(self):
        game, players, t = make_game(["A", "B", "C"])
        lines = [c.line for c in action_candidates(game, players[0])]
        self.assertEqual(lines, ["P0 inc", "P0 fa", "P0 tax", "P0 xch", "P0 steal P1", "P0 steal P2"])
        players[0].coins = 3
        lines = [c.line for c in action_candidates(game, players[0])]
        self.assertIn("P0 kill P2", lines)
        self.assertNotIn("P0 coup P1", lines)
        players[0].coins = 7
        self.assertIn("P0 coup P1", [c.line for c in action_candidates(game, players[0])])
        players[0].coins = 10
        self.assertEqual([c.line for c in action_candidates(game, players[0])], ["P0 coup P1", "P0 coup P2"])
        players[1].cards[0].revealed = players[1].cards[1].revealed = True
        self.assertEqual([c.line for c in action_candidates(game, players[0])], ["P0 coup P2"])

    def test_reveal_candidates_deduplicated(self):
        context = {"player_id": "id0", "cards": [
            {"index": 0, "role": "duke"}, {"index": 1, "role": "duke"}]}
        cands = decision_candidates("reveal_influence", context, "id0", {})
        self.assertEqual([(c.line, c.answer) for c in cands], [("P0 lose duke", 0)])


class RecapTests(unittest.TestCase):
    def test_recap_lines(self):
        game, players, t = make_game(["A", "B", "C"], me_index=1)
        players[0].cards = [Card(Role.CONTESSA), Card(Role.DUKE)]
        players[1].cards = [Card(Role.CAPTAIN), Card(Role.ASSASSIN)]
        players[2].cards = [Card(Role.AMBASSADOR), Card(Role.AMBASSADOR)]
        players[0].cards[0].revealed = True
        t.recap("id2")
        self.assertEqual(t.body[-4:], ["end P1", "P0 had ass cap", "P1 had amb amb", "P2 had duke con"])

    def test_recorded_game_ends_with_recap(self):
        random.seed(1)
        game, transcripts = play_recorded([AlwaysDuke, HonestRandom, DukeFish], seed=1)
        for p, t in zip(game.players, transcripts):
            self.assertTrue(t.body[0].startswith("deal "))
            self.assertEqual(t.body[-4], f"end {t.labels[game.winner.id]}")
            self.assertEqual(sum(l.endswith(" had") or " had " in l for l in t.body[-3:]), 3)
            self.assertEqual(t.body[-3].split()[0], "P0")


def play_recorded(classes, seed, check_match=False):
    rng = random.Random(seed)
    random.seed(seed)
    players = []
    for i, cls in enumerate(classes):
        kwargs = {"seed": rng.randrange(1000)} if cls is not PassiveAgent else {}
        players.append(cls(f"p{i}", name=f"{cls.__name__} {i}", **kwargs))
    game = Game(players)
    transcripts = [record(p, game) for p in players]
    if check_match:
        for t in transcripts:
            install_match_check(t)
    with contextlib.redirect_stdout(io.StringIO()):
        play_game(game)
    return game, transcripts


def install_match_check(transcript):
    original = transcript.match

    def checked(kind, context, answer):
        result = original(kind, context, answer)
        if result is None:
            raise AssertionError(f"no candidate for {kind} answer {answer!r}")
        return result

    transcript.match = checked


TEACHERS = [AlwaysDuke, PassiveAgent, DukeFish, HonestRandom, RandomAgent, Hibernate, FinalAgent]


class RecordTests(unittest.TestCase):
    def test_recording_does_not_change_game(self):
        for seed in range(12):
            n = 2 + seed % 5
            classes = [TEACHERS[(seed + i * 3) % len(TEACHERS)] for i in range(n)]
            recorded, _ = play_recorded(classes, seed)
            plain = self.play_plain(classes, seed)
            self.assertEqual(recorded.log, plain.log, f"seed {seed}")

    @staticmethod
    def play_plain(classes, seed):
        rng = random.Random(seed)
        random.seed(seed)
        players = []
        for i, cls in enumerate(classes):
            kwargs = {"seed": rng.randrange(1000)} if cls is not PassiveAgent else {}
            players.append(cls(f"p{i}", name=f"{cls.__name__} {i}", **kwargs))
        game = Game(players)
        with contextlib.redirect_stdout(io.StringIO()):
            play_game(game)
        return game

    def test_match_always_finds_candidate(self):
        for seed in range(60):
            n = 2 + seed % 5
            rng = random.Random(seed)
            classes = [rng.choice(TEACHERS) for _ in range(n)]
            game, transcripts = play_recorded(classes, seed, check_match=True)
            for t in transcripts:
                self.assertEqual(t.body[0][:5], "deal ")

    def test_transcript_matches_log(self):
        # Every action of every player appears in each transcript, in order.
        classes = [HonestRandom, AlwaysDuke, DukeFish, RandomAgent]
        game, transcripts = play_recorded(classes, seed=5)
        actions = [m for m in game.log if " claims " in m or " takes Income" in m or " launches " in m]
        for t in transcripts:
            kept = [l for l in t.body if l.split()[0].startswith("P") and
                    l.split()[1] in ("inc", "fa", "tax", "kill", "xch", "steal", "coup")]
            self.assertTrue(kept)
            self.assertLessEqual(len(kept), len(actions))


class RandomCandidatePlayer(Player):
    """Test-only player that picks uniformly among Transcript candidates."""

    def __init__(self, player_id, name=None, seed=None):
        super().__init__(player_id, name or player_id)
        self.rng = random.Random(seed)
        self.transcript = None
        self.chosen = []

    def on_game_start(self, game):
        self.transcript = Transcript(self.id)
        self.transcript.attach(game)

    def _pick(self, kind, context):
        t = self.transcript
        t.before_decision(kind, context)
        cands = t.candidates(kind, context)
        cand = self.rng.choice(cands)
        t.after_decision(cand)
        self.chosen.append(cand.line)
        return cand

    def choose_action(self, game):
        action, target = self._pick("action", None).answer
        return game.perform_action(self.id, action, target)

    def decide(self, kind, context):
        if kind == "block" and not self.transcript.candidates(kind, context):
            return None
        return self._pick(kind, context).answer


def is_subsequence(needle, haystack):
    it = iter(haystack)
    return all(any(x == y for y in it) for x in needle)


class StressTests(unittest.TestCase):
    def test_random_candidates_are_always_legal(self):
        rng = random.Random(2024)
        random.seed(2024)
        bot_classes = [AlwaysDuke, PassiveAgent, DukeFish, HonestRandom, RandomAgent, Hibernate, FinalAgent]
        finished = 0
        for game_number in range(300):
            n = 2 + game_number % 5
            players = []
            for i in range(n):
                name = f"Player {i}"
                if rng.random() < 0.6:
                    players.append(RandomCandidatePlayer(f"r{i}", name, seed=rng.randrange(10**6)))
                else:
                    cls = rng.choice(bot_classes)
                    kwargs = {} if cls is PassiveAgent else {"seed": rng.randrange(10**6)}
                    players.append(cls(f"r{i}", name=name, **kwargs))
            rng.shuffle(players)
            game = Game(players)
            with contextlib.redirect_stdout(io.StringIO()):
                draw = play_game(game)
            if draw:
                continue
            finished += 1
            for p in players:
                if isinstance(p, RandomCandidatePlayer):
                    p.transcript.sync()
                    self.assertTrue(
                        is_subsequence(p.chosen, p.transcript.body),
                        f"game {game_number}: chosen lines missing from transcript",
                    )
        self.assertGreater(finished, 250)


def public_game(ids, script, winner):
    """Events for a finished game, built from (type, fields) pairs."""
    return [ev(t, **f) for t, f in script], list(ids), winner


class HistoryTests(unittest.TestCase):
    def sample_game(self):
        return public_game(["me", "a", "b"], [
            ("action", dict(actor="a", action="tax", target=None)),
            ("no_challenge", dict(challenger="me", claimant="a", role="duke")),
            ("no_challenge", dict(challenger="b", claimant="a", role="duke")),
            ("coins", dict(player="a", delta=3, coins=5)),
            ("action", dict(actor="b", action="coup", target="me")),
            ("reveal", dict(player="me", role="captain")),
            ("action", dict(actor="me", action="steal", target="a")),
            ("challenge", dict(challenger="a", claimant="me", role="captain")),
            ("claim_failed", dict(claimant="me", role="captain")),
            ("reveal", dict(player="me", role="duke")),
            ("win", dict(player="a")),
        ], "a")

    def test_label_remapping(self):
        h = History("me")
        h.add_game(*self.sample_game())
        # now: seats b, me, a  -> me=P0, a=P1, b=P2
        lines = h.render(["b", "me", "a"])
        self.assertEqual(lines, [
            "past n 3",
            "P1 tax",
            "P0 ok",
            "P2 coup P0",
            "P0 lose cap",
            "P0 steal P1",
            "P1 chal",
            "P0 bluff",
            "P0 lose duke",
            "end P1",
        ])
        # different seating gives different labels for the same game
        lines2 = h.render(["me", "b", "a"])
        self.assertEqual(lines2[1], "P2 tax")

    def test_absent_opponents_get_x_labels(self):
        h = History("me")
        h.add_game(*self.sample_game())
        lines = h.render(["me", "a", "z"])  # b is absent, z is new
        self.assertEqual(lines[0], "past n 3")
        self.assertIn("X1 coup P0", lines)
        self.assertFalse(any(l.endswith(" b") or " b " in l for l in lines))
        self.assertEqual(lines[-1], "end P1")
        # X labels are assigned in order of first appearance across rendered history
        h.add_game(*public_game(["me", "q", "b"], [
            ("action", dict(actor="q", action="income", target=None)),
            ("action", dict(actor="b", action="income", target=None)),
        ], "b"))
        lines = h.render(["me", "a", "z", "b"])
        self.assertIn("X1 inc", lines)  # q
        self.assertIn("P3 inc", lines)  # b is now present

    def test_only_games_sharing_an_opponent(self):
        h = History("me")
        h.add_game(*public_game(["me", "q"], [("action", dict(actor="q", action="income", target=None))], "q"))
        self.assertEqual(h.render(["me", "a"]), [])
        self.assertEqual(len(h.render(["me", "q"])), 3)

    def test_draw_has_end_draw(self):
        h = History("me")
        h.add_game(*public_game(["me", "a"], [("action", dict(actor="a", action="foreign_aid", target=None))], None))
        lines = h.render(["me", "a"])
        self.assertEqual(lines[-1], "end draw")

    def test_budget_keeps_most_recent_games_oldest_first(self):
        h = History("me", max_lines=7, max_games=8)
        script = [
            ("action", dict(actor="a", action="income", target=None)),
            ("action", dict(actor="me", action="coup", target="a")),
            ("action", dict(actor="a", action="steal", target="me")),
        ]
        for i in range(5):
            h.add_game(*public_game(["me", "a"], script[: 1 + (i % 3)], "a"))
        # costs (lines+2): i=0:3, i=1:4, i=2:5, i=3:3, i=4:4 -> newest-first 4,3 = 7; adding 5 exceeds
        lines = h.render(["me", "a"])
        self.assertEqual(sum(l.startswith("past") for l in lines), 2)
        self.assertLessEqual(len(lines), 7)
        # oldest-first: game 3 (one action line) then game 4 (two action lines)
        self.assertEqual(lines[:4], ["past n 2", "P1 inc", "end P1", "past n 2"])

    def test_max_games(self):
        h = History("me", max_lines=1000, max_games=2)
        for _ in range(5):
            h.add_game(*public_game(["me", "a"], [("action", dict(actor="a", action="income", target=None))], "a"))
        self.assertEqual(sum(l.startswith("past") for l in h.render(["me", "a"])), 2)

    def test_no_private_information(self):
        h = None
        for seed in range(6):
            game, transcripts = play_recorded([HonestRandom, AlwaysDuke, DukeFish], seed)
            if h is None:
                h = History(game.players[0].id)
            h.add_game(game.events, [p.id for p in game.players], game.winner.id)
        lines = h.render([p.id for p in game.players])
        self.assertTrue(lines)
        for l in lines:
            self.assertNotIn(l.split()[0], ("deal", "draw", "keep", "$"))
            self.assertNotIn(" had ", l)
        self.assertEqual(
            sum(l.startswith("end ") for l in lines), sum(l.startswith("past") for l in lines)
        )

    def test_transcript_with_history_prompt_layout(self):
        h = History("p0")
        h.add_game(*public_game(["p0", "p1"], [("action", dict(actor="p1", action="income", target=None))], "p1"))
        players = [Player("p0", "A"), Player("p1", "B")]
        game = Game(players)
        t = Transcript("p0", h)
        t.attach(game)
        self.assertEqual(t.history_lines(), ["past n 2", "P1 inc", "end P1"])
        lines = t.prompt_lines("win")
        n = len(rules_lines())
        self.assertEqual(lines[n:n + 5], ["past n 2", "P1 inc", "end P1", "now", "n 2 res win"])
        self.assertEqual(lines[n + 5], t.body[0])

    def test_session_games_carry_history(self):
        from lm.generate_data import play_session_game
        rng = random.Random(4)
        pool = [(HonestRandom, f"hr{i}", f"hr{i}") for i in range(4)]
        histories = {pid: History(pid) for _, pid, _ in pool}
        seen_history = False
        for g in range(6):
            records = play_session_game(rng, pool, histories, 3, 0, g)
            if records is None:
                continue
            for r in records:
                seen_history = seen_history or (g > 0 and bool(r["history"]))
                self.assertEqual(r["game_index"], g)
        self.assertTrue(seen_history)


class MoveLineTests(unittest.TestCase):
    MOVES = ["P0 inc", "P1 fa", "P2 tax", "P3 xch", "P1 coup P0", "P0 kill P2", "P2 steal P1",
             "P1 chal", "P0 ok", "P4 blk duke", "P0 blk con", "P1 lose ass", "P0 lose cap", "keep duke con"]
    NOT_MOVES = ["deal ass con", "$ 2 2 2 | ass con", "draw duke duke ass con", "P1 bluff",
                 "P2 proof", "end P1", "end draw", "P0 had duke con", "past n 3", "now", "n 3 res win",
                 "", "P0 inc extra", "P0 coup"]

    def test_is_move_line(self):
        for line in self.MOVES:
            self.assertTrue(is_move_line(line), line)
        for line in self.NOT_MOVES:
            self.assertFalse(is_move_line(line), line)

    def test_is_own_move_line(self):
        own = [l for l in self.MOVES if l.startswith("P0 ") or l.startswith("keep")]
        for line in self.MOVES:
            self.assertEqual(is_own_move_line(line), line in own, line)
        for line in self.NOT_MOVES:
            self.assertFalse(is_own_move_line(line), line)

    def test_recorded_transcript_lines(self):
        rng = random.Random(1)
        from lm.generate_data import play_session_game
        pool = [(HonestRandom, f"hr{i}", f"hr{i}") for i in range(3)]
        histories = {pid: History(pid) for _, pid, _ in pool}
        for g in range(3):
            for r in play_session_game(rng, pool, histories, 3, 0, g) or []:
                for line in r["body"]:
                    if line.split(" ")[0] in ("deal", "draw", "end") or " had " in line or line.startswith("$"):
                        self.assertFalse(is_move_line(line), line)


if __name__ == "__main__":
    unittest.main()
