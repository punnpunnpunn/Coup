from __future__ import annotations

from dataclasses import dataclass

from agents.aggressive_probabilistic_belief import (
	AggressiveProbabilisticBelief,
)


@dataclass
class _ConfigurationResults:
	games: int = 0
	wins: int = 0

	@property
	def win_rate(self) -> float:
		return self.wins / self.games


class ProbabilisticBeliefTweaker(AggressiveProbabilisticBelief):
	"""Tune challenge and block thresholds from observed game wins."""

	GAMES_PER_CONFIGURATION = 100
	INITIAL_STEP_SIZE = 0.10
	MINIMUM_STEP_SIZE = 0.01

	def __init__(
		self,
		player_id: str,
		name: str | None = None,
		seed: int | None = None,
	):
		super().__init__(player_id, name, seed)
		self.games_played = 0
		self.wins = 0
		self._step_size = self.INITIAL_STEP_SIZE
		self._active_configuration = (
			self.challenge_probability,
			self.block_probability,
		)
		self._configuration_results: dict[
			tuple[float, float], _ConfigurationResults
		] = {}

	@property
	def win_rate(self) -> float:
		return self.wins / self.games_played if self.games_played else 0.0

	def on_game_end(self, winner_id: str | None) -> None:
		self.games_played += 1
		won = winner_id == self.id
		self.wins += int(won)

		results = self._configuration_results.setdefault(
			self._active_configuration, _ConfigurationResults()
		)
		results.games += 1
		results.wins += int(won)

		if results.games % self.GAMES_PER_CONFIGURATION == 0:
			self._select_next_configuration()

	def _select_next_configuration(self) -> None:
		best_configuration = max(
			self._configuration_results,
			key=lambda configuration: (
				self._configuration_results[configuration].win_rate,
				configuration == self._active_configuration,
				self._configuration_results[configuration].games,
			),
		)

		while True:
			candidates = self._neighboring_configurations(
				best_configuration, self._step_size
			)
			untried = [
				configuration
				for configuration in candidates
				if configuration not in self._configuration_results
			]
			if untried:
				next_configuration = untried[0]
				break

			if self._step_size <= self.MINIMUM_STEP_SIZE:
				next_configuration = min(
					candidates,
					key=lambda configuration: (
						self._configuration_results[configuration].games,
						self._configuration_results[configuration].win_rate,
					),
				)
				break

			self._step_size = max(
				self.MINIMUM_STEP_SIZE, self._step_size / 2
			)

		self._active_configuration = next_configuration
		self.challenge_probability, self.block_probability = (
			next_configuration
		)

	@staticmethod
	def _neighboring_configurations(
		configuration: tuple[float, float],
		step_size: float,
	) -> list[tuple[float, float]]:
		challenge_probability, block_probability = configuration
		neighbors = []
		for challenge_delta, block_delta in (
			(-step_size, 0.0),
			(step_size, 0.0),
			(0.0, -step_size),
			(0.0, step_size),
		):
			candidate = (
				round(
					min(1.0, max(0.0, challenge_probability + challenge_delta)),
					6,
				),
				round(
					min(1.0, max(0.0, block_probability + block_delta)),
					6,
				),
			)
			if candidate != configuration and candidate not in neighbors:
				neighbors.append(candidate)
		return neighbors
