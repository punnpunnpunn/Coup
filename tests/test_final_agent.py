import unittest
from unittest.mock import patch

from agents.aggressive_belief import AggressiveBelief
from agents.final_agent import (
	INITIAL_PROBABILISTIC_GAMES,
	FinalAgent,
)
from src.cards import Card, Role
from src.game import Game
from src.player import Player


class FinalAgentTests(unittest.TestCase):
	def setUp(self):
		self.agent = FinalAgent("p1", seed=7)
		self.opponent = Player("p2", "Opponent")
		self.third_player = Player("p3", "Third player")
		self.game = Game([self.agent, self.opponent, self.third_player])
		self.agent.cards = [Card(Role.CAPTAIN), Card(Role.ASSASSIN)]
		self.opponent.cards = [Card(Role.CONTESSA), Card(Role.AMBASSADOR)]
		self.third_player.cards = [Card(Role.CAPTAIN), Card(Role.CONTESSA)]
		self.agent.belief_tracker.sync_own_cards(
			[card.role for card in self.agent.cards]
		)
		self.agent._has_acquired_duke = False
		self.agent._use_probabilistic_duke_fish = False

	def test_does_not_inherit_aggressive_belief(self):
		self.assertFalse(issubclass(FinalAgent, AggressiveBelief))

	def test_fishes_for_duke_on_first_two_turns(self):
		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_count, 2)
		self.assertTrue(all(call.args[1] == "exchange" for call in perform.call_args_list))

	def test_stops_exchanging_after_two_duke_less_turns(self):
		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)
			self.agent.choose_action(self.game)
			self.agent.choose_action(self.game)

		self.assertEqual(
			[call.args[1] for call in perform.call_args_list],
			["exchange", "exchange", "income"],
		)

	def test_exchange_keeps_duke_fishing_preferences(self):
		self.agent._turns_exchange = 3
		cards = [
			{"index": index, "role": role.value}
			for index, role in enumerate(
				[Role.ASSASSIN, Role.DUKE, Role.CONTESSA, Role.AMBASSADOR]
			)
		]

		self.assertEqual(
			self.agent.decide("exchange", {"cards": cards, "keep_count": 2}),
			[1, 2],
		)

	def test_stops_fishing_as_soon_as_it_has_a_duke(self):
		self.agent.cards = [Card(Role.DUKE), Card(Role.ASSASSIN)]

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "tax")

	def test_takes_income_at_five_coins_when_opponents_have_at_most_three(self):
		self.agent.coins = 5
		self.agent._turns_exchange = 2
		self.opponent.coins = 3

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "income")

	def test_exchanges_at_five_coins_when_an_opponent_has_more_than_six(self):
		self.agent.coins = 5
		self.agent._turns_exchange = 2
		self.opponent.coins = 7

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "foreign_aid")

	def test_non_endgame_block_does_not_depend_on_aggressive_belief(self):
		self.agent.cards = [Card(Role.DUKE), Card(Role.ASSASSIN)]

		result = self.agent.decide(
			"block",
			{
				"eligible_blockers": [self.agent.id],
				"action": "foreign_aid",
			},
		)

		self.assertEqual(result, {"blocker_id": self.agent.id})

	def test_resets_exchange_turns_when_duke_claim_is_proven(self):
		self.agent._turns_exchange = 3

		self.agent.observe(
			"claim",
			{
				"claimant_id": self.opponent.id,
				"role": Role.DUKE.value,
				"proven": True,
			},
		)

		self.assertEqual(self.agent._turns_exchange, 0)

	def test_does_not_reset_exchange_turns_for_unproven_duke_claim(self):
		self.agent._turns_exchange = 3

		self.agent.observe(
			"claim",
			{
				"claimant_id": self.opponent.id,
				"role": Role.DUKE.value,
				"disproven": False,
			},
		)

		self.assertEqual(self.agent._turns_exchange, 3)

	def test_counts_reaching_endgame_once_per_game(self):
		self.agent._remaining_influences = 2

		with patch.object(self.agent, "_choose_endgame_action", return_value={}):
			self.agent.choose_action(self.game)
			self.agent.choose_action(self.game)

		self.assertEqual(self.agent._endgame_count, 1)

	def test_endgame_prefers_tax_over_exchange(self):
		self.agent._remaining_influences = 4
		self.agent.cards = [Card(Role.DUKE), Card(Role.AMBASSADOR)]

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "tax")

	def test_endgame_uses_exchange_after_higher_priority_actions(self):
		self.agent._remaining_influences = 4
		self.agent.cards = [Card(Role.AMBASSADOR), Card(Role.CONTESSA)]

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "exchange")

	def test_resets_endgame_reached_flag_for_new_game(self):
		self.agent._reach_endgame = True

		self.agent.observe("new_game", {"player_count": 3})

		self.assertFalse(self.agent._reach_endgame)

	def test_uses_probabilistic_duke_fish_during_initial_games(self):
		self.agent._use_probabilistic_duke_fish = True
		self.agent.cards = [Card(Role.ASSASSIN), Card(Role.CAPTAIN)]

		with patch.object(self.game, "perform_action", return_value={}) as perform:
			self.agent.choose_action(self.game)

		self.assertEqual(perform.call_args.args[1], "exchange")

	def test_switches_to_final_strategy_if_first_50_win_rate_is_below_25_percent(self):
		self.agent._use_probabilistic_duke_fish = True
		for game_index in range(INITIAL_PROBABILISTIC_GAMES):
			winner_id = self.agent.id if game_index < 12 else self.opponent.id
			self.agent.on_game_end(winner_id)

		self.assertFalse(self.agent._use_probabilistic_duke_fish)

	def test_counts_a_draw_as_a_played_non_win(self):
		self.agent._use_probabilistic_duke_fish = True

		self.agent.on_game_end(None)

		self.assertEqual(self.agent._probabilistic_games_played, 1)
		self.assertEqual(self.agent._probabilistic_wins, 0)

	def test_keeps_probabilistic_strategy_if_first_50_win_rate_exceeds_25_percent(self):
		self.agent._use_probabilistic_duke_fish = True
		for game_index in range(INITIAL_PROBABILISTIC_GAMES):
			winner_id = self.agent.id if game_index < 13 else self.opponent.id
			self.agent.on_game_end(winner_id)

		self.assertTrue(self.agent._use_probabilistic_duke_fish)

	def test_switches_if_probabilistic_win_rate_later_dips_below_25_percent(self):
		self.agent._use_probabilistic_duke_fish = True
		for game_index in range(INITIAL_PROBABILISTIC_GAMES):
			winner_id = self.agent.id if game_index < 13 else self.opponent.id
			self.agent.on_game_end(winner_id)

		self.agent.on_game_end(self.opponent.id)
		self.assertTrue(self.agent._use_probabilistic_duke_fish)
		self.agent.on_game_end(self.opponent.id)
		self.assertTrue(self.agent._use_probabilistic_duke_fish)
		self.agent.on_game_end(self.opponent.id)
		self.assertFalse(self.agent._use_probabilistic_duke_fish)

	def test_probabilistic_duke_fish_uses_probabilistic_challenges(self):
		self.agent._use_probabilistic_duke_fish = True

		with patch.object(
			self.agent.belief_tracker,
			"role_probability",
			return_value=0.01,
		):
			self.assertTrue(
				self.agent.decide(
					"challenge",
					{
						"claimant_id": self.opponent.id,
						"role": Role.DUKE.value,
					},
				)
			)


if __name__ == "__main__":
	unittest.main()
