import tempfile
import unittest
from pathlib import Path

from src.cards import Card, Role
from src.game import Game
from src.player import Player
from submission import SubmissionAgent, validate_source


class NonChallengingPlayer(Player):
    def decide(self, kind, context):
        if kind == "challenge":
            return False
        if kind == "block":
            return None
        raise AssertionError(f"Unexpected decision: {kind}")


class SubmissionTests(unittest.TestCase):
    def test_source_validator_allows_standard_library_and_rejects_external(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "agent.py"
            source.write_text("import json\n", encoding="utf-8")
            self.assertEqual(validate_source(source), source.resolve())

            source.write_text("import requests\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "standard-library"):
                validate_source(source)

    def test_starter_agent_completes_action_and_persists_memory(self):
        source = Path(__file__).parents[1] / "submission/agents/starter_agent.py"
        agent = SubmissionAgent("agent", "Starter", source)
        opponent = Player("opponent", "Opponent")

        first_game = Game([agent, opponent])
        agent.choose_action(first_game)
        self.assertEqual(agent.memory, {"turns": 1})

        second_game = Game([agent, opponent])
        agent.choose_action(second_game)
        self.assertEqual(agent.memory, {"turns": 2})

    def test_illegal_action_forfeits_and_advances_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "illegal.py"
            source.write_text(
                'import json, sys\n'
                'json.load(sys.stdin)\n'
                'print(json.dumps({"action": {"name": "invalid", '
                '"target_id": None}, "memory": None}))\n',
                encoding="utf-8",
            )
            agent = SubmissionAgent("agent", "Illegal", source)
            opponent = Player("opponent", "Opponent")
            game = Game([agent, opponent])

            agent.choose_action(game)

        self.assertFalse(agent.alive)
        self.assertEqual(game.disqualifications, {"agent": "invalid_response"})
        self.assertEqual(game.current.id, "opponent")

    def test_submission_receives_own_decision_context(self):
        source = Path(__file__).parents[1] / "submission/agents/starter_agent.py"
        agent = SubmissionAgent("agent", "Starter", source)
        opponent = Player("opponent", "Opponent")
        game = Game([opponent, agent])

        decision = game._decide(
            None,
            "challenge",
            {
                "decision_player_id": agent.id,
                "challenger_id": agent.id,
                "claimant_id": opponent.id,
                "role": "duke",
            },
        )

        self.assertIs(decision, False)
        self.assertIsNone(agent._active_game)

    def test_timeout_limit_must_be_less_than_two_seconds(self):
        source = Path(__file__).parents[1] / "submission/agents/starter_agent.py"
        with self.assertRaisesRegex(ValueError, "at most"):
            SubmissionAgent(
                "agent",
                "Starter",
                source,
                timeout_seconds=2.0,
            )

    def test_timed_out_agent_is_forfeited(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "slow.py"
            source.write_text(
                "import time\n"
                "time.sleep(1)\n",
                encoding="utf-8",
            )
            agent = SubmissionAgent(
                "agent",
                "Slow",
                source,
                timeout_seconds=0.1,
            )
            game = Game([agent, Player("opponent", "Opponent")])

            agent.choose_action(game)

        self.assertEqual(game.disqualifications["agent"], "timeout")
        self.assertFalse(agent.alive)

    def test_illegal_exchange_response_does_not_restore_forfeited_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_exchange.py"
            source.write_text(
                'import json, sys\n'
                'request = json.load(sys.stdin)\n'
                'if request["phase"] == "action":\n'
                '    action = next(a for a in request["legal_actions"] '
                'if a["name"] == "exchange")\n'
                '    result = {"action": action, "memory": None}\n'
                'else:\n'
                '    result = {"decision": [999], "memory": None}\n'
                'print(json.dumps(result))\n',
                encoding="utf-8",
            )
            agent = SubmissionAgent("agent", "Bad exchange", source)
            opponent = NonChallengingPlayer("opponent", "Opponent")
            game = Game([agent, opponent])
            agent.cards = [Card(Role.AMBASSADOR), Card(Role.DUKE)]

            agent.choose_action(game)

        self.assertFalse(agent.alive)
        self.assertEqual(game.disqualifications["agent"], "invalid_response")
        self.assertIs(game.winner, opponent)


if __name__ == "__main__":
    unittest.main()
