import math
import random

from lm.encoding import History, Transcript
from src.game import Game
from src.player import Player

INSTALL_HINT = (
	"The LLM agent needs torch and transformers: pip install -r requirements-llm.txt"
)


class LLMAgent(Player):
	"""Plays by scoring every legal move with a fine-tuned language model."""

	def __init__(
		self,
		player_id: str,
		name: str | None = None,
		seed: int | None = None,
		model_path: str | None = None,
		temperature: float = 1.0,
		greedy: bool = False,
		scorer=None,
		history_max_lines: int = 600,
		on_decision=None,
	):
		super().__init__(player_id, name or player_id)
		self.rng = random.Random(seed)
		self.model_path = model_path
		self.temperature = temperature
		self.greedy = greedy
		self._scorer = scorer
		# optional callback(prefix_lines, candidate_lines, chosen_index) for RL
		self.on_decision = on_decision
		self._game: Game | None = None
		self._transcript: Transcript | None = None
		self.history = History(player_id, max_lines=history_max_lines)
		self._game_open = False
		self._session = None  # incremental KV cache, created lazily if the scorer supports it

	# ------------------------------------------------------------------
	def _get_scorer(self):
		if self._scorer is None:
			try:
				from lm.model import CoupLM
			except ImportError as error:
				raise RuntimeError(f"{INSTALL_HINT} ({error})") from error
			self._scorer = CoupLM.load(self.model_path)
		return self._scorer

	def _archive_game(self, winner_id) -> None:
		game = self._game
		if game is not None and self._game_open:
			self.history.add_game(
				list(game.events), [p.id for p in game.players], winner_id
			)
		self._game_open = False

	def on_game_start(self, game: Game) -> None:
		# previous game never finished (draw / turn cap): store it as a draw
		self._archive_game(None)
		self._game = game
		self._transcript = None
		self._game_open = True
		if self._session is not None:
			self._session.reset()  # weights may have changed between games
		self._ensure_transcript(game)

	def on_game_end(self, winner_id: str) -> None:
		self._archive_game(winner_id)

	def _ensure_transcript(self, game) -> Transcript | None:
		if game is not None and game is not self._game:
			self._game = game
			self._transcript = None
			self._game_open = True
		if self._transcript is None and self._game is not None and self.cards:
			transcript = Transcript(self.id, history=self.history)
			transcript.attach(self._game)
			self._transcript = transcript
		return self._transcript

	def _pick(self, kind: str, context, game) -> object:
		transcript = self._ensure_transcript(game)
		if transcript is None:
			raise RuntimeError("LLMAgent cannot find the game it is playing in.")
		candidates = transcript.candidates(kind, context)
		if not candidates:
			return None
		transcript.before_decision(kind, context)
		if len(candidates) == 1:
			choice = candidates[0]
		else:
			prefix = transcript.prompt_lines("win")
			lines = [c.line for c in candidates]
			scorer = self._get_scorer()
			if self._session is None and hasattr(scorer, "new_session"):
				self._session = scorer.new_session()
			if self._session is not None:
				scores = scorer.score(prefix, lines, session=self._session)
			else:
				scores = scorer.score(prefix, lines)
			index = self._select(scores)
			choice = candidates[index]
			if self.on_decision is not None:
				self.on_decision(prefix, lines, index)
		transcript.after_decision(choice)
		return choice.answer

	def _select(self, scores: list[float]) -> int:
		if self.greedy or self.temperature <= 0:
			return max(range(len(scores)), key=lambda i: scores[i])
		top = max(scores)
		weights = [math.exp((s - top) / self.temperature) for s in scores]
		return self.rng.choices(range(len(scores)), weights=weights)[0]

	# ------------------------------------------------------------------
	def choose_action(self, game: Game) -> dict:
		if game.current.id != self.id:
			raise ValueError("It is not this agent's turn.")
		action, target_id = self._pick("action", None, game)
		return game.perform_action(self.id, action, target_id)

	def decide(self, kind: str, context: dict[str, object]) -> object:
		return self._pick(kind, context, self._game)
