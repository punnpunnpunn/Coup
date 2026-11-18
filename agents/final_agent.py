import random

from agents.aggressive_probabilistic_belief import (
	CHALLENGE_ROLE_PROBABILITY,
)
from agents.probabilistic_belief import ProbabilisticBeliefTracker
from src.cards import Role
from src.game import Game
from src.player import Player


INITIAL_PROBABILISTIC_GAMES = 50
MINIMUM_PROBABILISTIC_WIN_RATE = 0.25


class FinalAgent(Player):
	"""Start with probabilistic Duke fishing, then select a strategy by win rate."""

	def __init__(
		self,
		player_id: str,
		name: str | None = None,
		seed: int | None = None,
	):
		super().__init__(player_id, name or player_id)
		self.rng = random.Random(seed)
		self.belief_tracker = ProbabilisticBeliefTracker(player_id)
		self._game: Game | None = None
		self._remaining_influences: int | None = None
		self._has_acquired_duke = False
		self._turns_exchange = 0
		self._reach_endgame = False
		self._endgame_count = 0
		self._probabilistic_games_played = 0
		self._probabilistic_wins = 0
		self._use_probabilistic_duke_fish = True

	def observe(self, kind: str, context: dict[str, object]) -> None:
		self.belief_tracker.observe(kind, context)
		if kind == "initial_hand" or (
			kind == "exchange" and context.get("player_id") == self.id
		) or (kind == "reveal" and context.get("player_id") == self.id) or (
			kind == "claim" and context.get("claimant_id") == self.id
		):
			self.belief_tracker.sync_own_cards(
				[card.role for card in self.cards]
			)

		if kind == "new_game":
			player_count = context["player_count"]
			if not isinstance(player_count, int):
				raise ValueError("New game observation has an invalid player count.")
			self._remaining_influences = player_count * 2 - 2
			self._has_acquired_duke = False
			self._turns_exchange = 0
			self._reach_endgame = False
			return

		if kind == "initial_hand" and context.get("player_id") == self.id:
			self._has_acquired_duke |= any(
				card.role == Role.DUKE for card in self.cards
			)
		elif kind == "exchange" and context.get("player_id") == self.id:
			self._has_acquired_duke |= any(
				card.role == Role.DUKE and not card.revealed
				for card in self.cards
			)
		elif kind == "reveal":
			if context.get("player_id") == self.id:
				self._has_acquired_duke |= any(
					card.role == Role.DUKE and not card.revealed
					for card in self.cards
				)
			elif self._remaining_influences is not None:
				self._remaining_influences -= 1
		elif (
			kind == "claim"
			and context.get("role") == Role.DUKE.value
			and context.get("proven") is True
		):
			self._turns_exchange = 0

	def on_game_end(self, winner_id: str | None) -> None:
		if not self._use_probabilistic_duke_fish:
			return

		self._probabilistic_games_played += 1
		self._probabilistic_wins += int(winner_id == self.id)
		win_rate = (
			self._probabilistic_wins / self._probabilistic_games_played
		)

		if self._probabilistic_games_played == INITIAL_PROBABILISTIC_GAMES:
			self._use_probabilistic_duke_fish = (
				win_rate > MINIMUM_PROBABILISTIC_WIN_RATE
			)
		elif (
			self._probabilistic_games_played > INITIAL_PROBABILISTIC_GAMES
			and win_rate < MINIMUM_PROBABILISTIC_WIN_RATE
		):
			self._use_probabilistic_duke_fish = False

	def choose_action(self, game: Game) -> dict:
		if game.current.id != self.id:
			raise ValueError("It is not this agent's turn.")
		self._game = game
		self._turns_exchange += 1
		self._has_acquired_duke |= any(
			card.role == Role.DUKE for card in self.cards
		)

		if self._is_endgame and not self._reach_endgame:
			self._reach_endgame = True
			self._endgame_count += 1
		if self._use_probabilistic_duke_fish:
			return self._choose_probabilistic_duke_fish_action(game)
		if self._is_endgame:
			return self._choose_endgame_action(game)

		if self.coins >= 7:
			opponents = [
				player for player in game.alive_players if player.id != self.id
			]
			highest_coin_total = max(player.coins for player in opponents)
			top_coin_opponents = [
				player for player in opponents
				if player.coins == highest_coin_total
			]
			return game.perform_action(
				self.id, "coup", target_id=self.rng.choice(top_coin_opponents).id
			)

		if any(
			card.role == Role.DUKE and not card.revealed for card in self.cards
		):
			return game.perform_action(self.id, "tax")

		revealed_dukes = sum(
			card.role == Role.DUKE and card.revealed
			for player in game.players
			for card in player.cards
		)
		if revealed_dukes >= 3:
			return game.perform_action(self.id, "foreign_aid")

		if not self._has_acquired_duke and self._turns_exchange <= 2:
			return game.perform_action(self.id, "exchange")
		if self.coins != 5:
			return game.perform_action(self.id, "income")
		opponents = [
			player for player in game.alive_players if player.id != self.id
		]
		if self.coins == 5 and all(player.coins <= 6 for player in opponents):
			return game.perform_action(self.id, "income")
		return game.perform_action(self.id, "foreign_aid")

	def decide(self, kind: str, context: dict[str, object]) -> object:
		if self._use_probabilistic_duke_fish:
			return self._decide_probabilistic_duke_fish(kind, context)
		if self._is_endgame:
			return self._decide_endgame(kind, context)

		if kind == "challenge":
			return False
		if kind == "block":
			return self._decide_block(context)
		if kind == "reveal_influence":
			return context["cards"][0]["index"]
		if kind == "exchange":
			return self._choose_fish_exchange(
				context["cards"], context["keep_count"]
			)
		raise ValueError(f"Unknown decision type: {kind}")

	def _choose_probabilistic_duke_fish_action(self, game: Game) -> dict:
		roles = {card.role for card in self.cards if not card.revealed}
		if self.coins >= 7:
			opponents = [
				player for player in game.alive_players if player.id != self.id
			]
			top_coins = max(player.coins for player in opponents)
			targets = [
				player for player in opponents if player.coins == top_coins
			]
			return game.perform_action(
				self.id, "coup", target_id=self.rng.choice(targets).id
			)

		if Role.DUKE in roles:
			return game.perform_action(self.id, "tax")

		revealed_dukes = sum(
			card.role == Role.DUKE and card.revealed
			for player in game.players
			for card in player.cards
		)
		action = "foreign_aid" if revealed_dukes >= 3 else "exchange"
		return game.perform_action(self.id, action)

	def _decide_probabilistic_duke_fish(
		self, kind: str, context: dict[str, object]
	) -> object:
		if kind == "challenge":
			return (
				self.belief_tracker.role_probability(
					context["claimant_id"], context["role"]
				)
				<= CHALLENGE_ROLE_PROBABILITY
			)

		if kind == "reveal_influence":
			cards = context["cards"]
			non_dukes = [
				card for card in cards if card["role"] != Role.DUKE.value
			]
			return (non_dukes or cards)[0]["index"]

		if kind == "exchange":
			return self._choose_fish_exchange(
				context["cards"], context["keep_count"]
			)

		if kind == "block":
			decision = self._decide_block(context)
			if (
				decision is None
				and context["action"] == "assassinate"
				and self.id in context["eligible_blockers"]
				and self.influence == 1
			):
				return {
					"blocker_id": self.id,
					"role": Role.CONTESSA.value,
				}
			return decision

		raise ValueError(f"Unknown decision type: {kind}")

	def _choose_endgame_action(self, game: Game) -> dict:
		roles = {card.role for card in self.cards if not card.revealed}
		opponents = [
			player for player in game.alive_players if player.id != self.id
		]

		if self.coins >= 7:
			target = max(opponents, key=lambda player: player.influence)
			return game.perform_action(self.id, "coup", target_id=target.id)

		if Role.ASSASSIN in roles and self.coins >= 3:
			targets = [
				player
				for player in opponents
				if self._cannot_block(player, {Role.CONTESSA}, game)
			]
			if targets:
				target = min(targets, key=lambda player: player.influence)
				return game.perform_action(
					self.id, "assassinate", target_id=target.id
				)

		if Role.DUKE in roles:
			return game.perform_action(self.id, "tax")

		if Role.CAPTAIN in roles:
			targets = [
				player
				for player in opponents
				if self._cannot_block(
					player, {Role.AMBASSADOR, Role.CAPTAIN}, game
				)
			]
			if targets:
				target = max(targets, key=lambda player: player.coins)
				return game.perform_action(self.id, "steal", target_id=target.id)

		if all(
			self._cannot_block(player, {Role.DUKE}, game)
			for player in opponents
		):
			return game.perform_action(self.id, "foreign_aid")

		if Role.AMBASSADOR in roles:
			return game.perform_action(self.id, "exchange")

		return game.perform_action(self.id, "income")

	def _decide_endgame(
		self, kind: str, context: dict[str, object]
	) -> object:
		if kind == "challenge":
			return self.belief_tracker.contradicts(
				context["claimant_id"], context["role"]
			)

		if kind == "block":
			return self._decide_block(context)

		if kind == "reveal_influence":
			return self._ranked_card_indices(context["cards"])[-1]

		if kind == "exchange":
			cards = context["cards"]
			keep_count = context["keep_count"]
			ranked_indices = self._ranked_card_indices(cards)
			kept: list[int] = []
			kept_roles: set[Role] = set()
			for index in ranked_indices:
				role = Role(cards[index]["role"])
				if role not in kept_roles:
					kept.append(index)
					kept_roles.add(role)
					if len(kept) == keep_count:
						return kept
			kept.extend(index for index in ranked_indices if index not in kept)
			return kept[:keep_count]

		raise ValueError(f"Unknown decision type: {kind}")

	def _decide_block(self, context: dict[str, object]) -> object:
		roles = {card.role.value for card in self.cards if not card.revealed}
		if self.id not in context["eligible_blockers"]:
			return None

		if context["action"] == "foreign_aid":
			if Role.DUKE.value not in roles:
				return None
			return {"blocker_id": self.id}

		claimable_roles = [
			role for role in context["allowed_roles"] if role in roles
		]
		if not claimable_roles:
			return None

		return {
			"blocker_id": self.id,
			"role": self.rng.choice(claimable_roles),
		}

	def _ranked_card_indices(
		self, cards: list[dict[str, object]]
	) -> list[int]:
		priorities = {
			Role.DUKE: 0,
			Role.ASSASSIN: 1,
			Role.CAPTAIN: 2,
			Role.CONTESSA: 3,
			Role.AMBASSADOR: 4,
		}
		return sorted(
			range(len(cards)),
			key=lambda index: (
				self._role_priority(Role(cards[index]["role"]), priorities),
				index,
			),
		)

	def _role_priority(
		self, role: Role, priorities: dict[Role, int]
	) -> int:
		if role in (Role.ASSASSIN, Role.CAPTAIN) and not self._can_use_role(
			role
		):
			return len(priorities) + priorities[role]
		return priorities[role]

	def _can_use_role(self, role: Role) -> bool:
		if role not in (Role.ASSASSIN, Role.CAPTAIN):
			return True
		if role == Role.ASSASSIN and self.coins < 3:
			return False
		if self._game is None:
			return False

		blocking_roles = (
			{Role.CONTESSA}
			if role == Role.ASSASSIN
			else {Role.AMBASSADOR, Role.CAPTAIN}
		)
		return any(
			player.id != self.id
			and player.alive
			and self._cannot_block(player, blocking_roles, self._game)
			for player in self._game.alive_players
		)

	def _cannot_block(
		self, target: Player, blocking_roles: set[Role], game: Game
	) -> bool:
		active_beliefs = list(
			self.belief_tracker.beliefs.get(target.id, [])
		)
		for card in target.cards:
			if card.revealed and card.role in active_beliefs:
				active_beliefs.remove(card.role)

		if (
			len(active_beliefs) >= target.influence
			and not blocking_roles.intersection(active_beliefs)
		):
			return True

		if blocking_roles.intersection(active_beliefs):
			return False

		return all(
			sum(
				self.belief_tracker.beliefs.get(player.id, []).count(role)
				for player in game.players
				if player.id != target.id
			)
			>= 3
			for role in blocking_roles
		)

	@property
	def _is_endgame(self) -> bool:
		return (
			self._remaining_influences is not None
			and self._remaining_influences <= 2
		)

	def _choose_fish_exchange(
		self, cards: list[dict[str, object]], keep_count: int
	) -> list[int]:
		role_indices = {
			role: [card["index"] for card in cards if card["role"] == role.value]
			for role in Role
		}
		dukes = role_indices[Role.DUKE]
		contessas = role_indices[Role.CONTESSA]
		ambassadors = role_indices[Role.AMBASSADOR]

		preferred: list[int] = []
		if keep_count >= 2 and dukes and contessas:
			preferred = [dukes[0], contessas[0]]
		elif keep_count >= 2 and dukes and ambassadors:
			preferred = [dukes[0], ambassadors[0]]
		elif keep_count >= 2 and not dukes and contessas and ambassadors:
			preferred = [contessas[0], ambassadors[0]]
		elif dukes:
			preferred = [dukes[0]]
		elif contessas or ambassadors:
			preferred = [ambassadors[0]] if ambassadors else [contessas[0]]

		remaining = [
			card["index"]
			for card in cards
			if card["index"] not in preferred
		]
		return preferred + self.rng.sample(
			remaining, keep_count - len(preferred)
		)
