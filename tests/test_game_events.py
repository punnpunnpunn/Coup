import contextlib
import io
import random
import unittest

from agents.always_duke import AlwaysDuke
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.honest_random import HonestRandom
from agents.random_agent import RandomAgent
from src.cards import Card, Role
from src.game import Game
from src.player import Player


class StartHookPlayer(Player):
    def on_game_start(self, game):
        self.seen = (game, [len(p.cards) for p in game.players], len(game.events))


class GameEventsTests(unittest.TestCase):
    def test_on_game_start_called_after_dealing(self):
        players = [StartHookPlayer("a", "A"), StartHookPlayer("b", "B"), Player("c", "C")]
        game = Game(players)
        for p in players[:2]:
            seen_game, card_counts, event_count = p.seen
            self.assertIs(seen_game, game)
            self.assertEqual(card_counts, [2, 2, 2])
            self.assertEqual(event_count, 0)

    def test_events_aligned_with_log_in_bot_games(self):
        bots = [AlwaysDuke, DukeFish, HonestRandom, RandomAgent, FinalAgent]
        seen_types = set()
        for seed in range(40):
            rng = random.Random(seed)
            random.seed(seed)
            n = 2 + seed % 5
            players = [
                rng.choice(bots)(f"p{i}", name=f"Bot {i}", seed=rng.randrange(1000))
                for i in range(n)
            ]
            game = Game(players)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(300):
                    if game.winner is not None:
                        break
                    game.current.choose_action(game)
            self.assertEqual(len(game.events), len(game.log))
            ids = {p.id for p in players}
            for event in game.events:
                seen_types.add(event["type"])
                for key in ("actor", "target", "player", "challenger", "claimant", "blocker"):
                    if event.get(key) is not None:
                        self.assertIn(event[key], ids)
        self.assertTrue(
            {"action", "coins", "challenge", "no_challenge", "claim_failed", "claim_proven",
             "replacement", "block", "block_result", "no_block", "reveal", "turn", "win",
             "exchanged", "action_failed"} <= seen_types,
            seen_types,
        )

    def test_event_contents(self):
        a, b = Player("a", "Same"), Player("b", "Same")
        game = Game([a, b])
        a.cards = [Card(Role.DUKE), Card(Role.CAPTAIN)]
        b.cards = [Card(Role.CONTESSA), Card(Role.CONTESSA)]

        def decisions(kind, context):
            if kind == "challenge":
                return False
            if kind == "block":
                return None
            raise AssertionError(kind)

        game.perform_action("a", "tax", decision_provider=decisions)
        game.perform_action("b", "income", decision_provider=decisions)
        game.perform_action("a", "steal", "b", decision_provider=decisions)
        self.assertEqual(game.events[0], {"type": "action", "actor": "a", "action": "tax", "target": None})
        self.assertIn(
            {"type": "no_challenge", "challenger": "b", "claimant": "a", "role": "duke"}, game.events
        )
        self.assertIn({"type": "coins", "player": "a", "delta": 3, "coins": 5}, game.events)
        self.assertIn({"type": "turn", "player": "b"}, game.events)
        self.assertIn({"type": "no_block", "action": "steal"}, game.events)
        steal = [e for e in game.events if e["type"] == "coins" and e.get("from")]
        self.assertEqual(steal, [{"type": "coins", "player": "a", "delta": 2, "coins": 7, "from": "b"}])
        self.assertEqual(len(game.events), len(game.log))

    def test_rollback_truncates_events(self):
        a, b = Player("a", "A"), Player("b", "B")
        game = Game([a, b])
        a.cards = [Card(Role.DUKE), Card(Role.CAPTAIN)]
        game.perform_action("a", "income")
        before_log, before_events = list(game.log), list(game.events)

        def bad(kind, context):
            if kind == "challenge":
                return False
            raise ValueError("boom")

        game.current_player = 1
        with self.assertRaises(ValueError):
            game.perform_action("b", "foreign_aid", decision_provider=bad)
        self.assertEqual(game.log, before_log)
        self.assertEqual(game.events, before_events)
        self.assertEqual(len(game.events), len(game.log))

    def test_invalid_action_leaves_events_untouched(self):
        a, b = Player("a", "A"), Player("b", "B")
        game = Game([a, b])
        with self.assertRaises(ValueError):
            game.perform_action("a", "coup", "b")
        self.assertEqual(game.events, [])
        self.assertEqual(game.log, [])


if __name__ == "__main__":
    unittest.main()
