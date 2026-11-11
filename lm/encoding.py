"""Text encoding of Coup games for the language-model agent.

Pure Python: no torch / transformers imports. Every transcript is written
from one player's perspective (P0 = that player, P1 = next seat, ...).
Past games (public events only) can be prepended via `History`.
"""
import itertools
import re
from typing import NamedTuple

ROLE_ABBR = {
	"duke": "duke",
	"assassin": "ass",
	"captain": "cap",
	"ambassador": "amb",
	"contessa": "con",
}
ABBR_ROLE = {abbr: role for role, abbr in ROLE_ABBR.items()}
_ROLE_ORDER = {abbr: i for i, abbr in enumerate(ROLE_ABBR.values())}

RULES = (
	"coup. roles duke ass cap amb con, 3 of each. each player: 2 cards, 2 coins. last alive wins.\n"
	"inc +1. fa +2, duke blocks. coup pay 7 target loses card, must coup at 10+.\n"
	"tax: duke +3. kill: ass pay 3 target loses card, con blocks. steal: cap take 2, cap/amb block. xch: amb draw 2 keep same count.\n"
	"any claim may be a bluff. chal: bluffer loses card, else challenger loses card.\n"
	"P0 is me."
)


_MOVE_RE = re.compile(
	r"^P\d+ (?:inc|fa|tax|xch|chal|ok|(?:coup|kill|steal) P\d+|(?:blk|lose) [a-z]+)$"
)


def is_move_line(line: str) -> bool:
	"""True for a player choice: `P# inc|fa|tax|xch|coup T|kill T|steal T|chal|ok|blk R|lose R`
	or `keep ...` (P0's exchange choice). Not deal/$/draw/bluff/proof/end/had/past/now lines."""
	return line.startswith("keep ") or _MOVE_RE.match(line) is not None


def is_own_move_line(line: str) -> bool:
	"""A move line made by P0 (including `keep ...`)."""
	return line.startswith("keep ") or (line.startswith("P0 ") and _MOVE_RE.match(line) is not None)


def rules_lines() -> list[str]:
	return RULES.split("\n")


def result_line(n_players: int, goal: str) -> str:
	return f"n {n_players} res {goal}"


def header_lines(n_players: int, goal: str) -> list[str]:
	"""RULES + the `n .. res ..` line (no history). Prompts are built by Transcript.prompt_lines."""
	return rules_lines() + [result_line(n_players, goal)]


class Candidate(NamedTuple):
	line: str
	answer: object
	synthetic: bool


def _canon(abbrs) -> list[str]:
	return sorted(abbrs, key=lambda a: _ROLE_ORDER[a])


def _roles_of(cards, include_revealed: bool = False) -> list[str]:
	return _canon(
		ROLE_ABBR[c.role.value] for c in cards if include_revealed or not c.revealed
	)


_ACTION_ALIASES = {
	"duke": "tax",
	"assassin": "assassinate",
	"ambassador": "exchange",
	"captain": "steal",
}
_ACTION_LINE = {
	"income": "inc",
	"foreign_aid": "fa",
	"tax": "tax",
	"exchange": "xch",
	"steal": "steal",
	"assassinate": "kill",
	"coup": "coup",
}


def normalize_action(action: str) -> str:
	action = action.strip().lower().replace(" ", "_")
	return _ACTION_ALIASES.get(action, action)


def make_labels(game, me_id: str) -> dict[str, str]:
	"""player_id -> relative seat label (P0 = me)."""
	return labels_for_ids([p.id for p in game.players], me_id)


def labels_for_ids(ids: list[str], me_id: str) -> dict[str, str]:
	mi = ids.index(me_id)
	return {pid: f"P{(i - mi) % len(ids)}" for i, pid in enumerate(ids)}


def translate_event(event: dict, label_of_id, me_id: str) -> str | None:
	"""Turn one public game event into a transcript line (None = dropped)."""
	L = label_of_id
	kind = event["type"]
	if kind == "action":
		line = _ACTION_LINE.get(event["action"])
		if line is None:
			return None
		text = f"{L[event['actor']]} {line}"
		if event["target"] is not None:
			text += f" {L[event['target']]}"
		return text
	if kind == "challenge":
		return f"{L[event['challenger']]} chal"
	if kind == "no_challenge":
		return "P0 ok" if event["challenger"] == me_id else None
	if kind == "claim_failed":
		return f"{L[event['claimant']]} bluff"
	if kind == "claim_proven":
		return f"{L[event['claimant']]} proof"
	if kind == "block":
		return f"{L[event['blocker']]} blk {ROLE_ABBR[event['role']]}"
	if kind == "reveal":
		return f"{L[event['player']]} lose {ROLE_ABBR[event['role']]}"
	return None


class _HistoryLabels:
	"""Labels for rendering past games: current labels, X1, X2, ... for absent players."""

	def __init__(self, current: dict[str, str]):
		self.current = current
		self.absent: dict[str, str] = {}

	def __getitem__(self, pid: str) -> str:
		if pid in self.current:
			return self.current[pid]
		if pid not in self.absent:
			self.absent[pid] = f"X{len(self.absent) + 1}"
		return self.absent[pid]


class History:
	"""Public events of finished games, re-rendered with the current game's labels."""

	def __init__(self, me_id: str, max_lines: int = 600, max_games: int = 8):
		self.me_id = me_id
		self.max_lines = max_lines
		self.max_games = max_games
		self.games: list[dict] = []

	def add_game(self, events: list[dict], player_ids: list[str], winner_id: str | None) -> None:
		self.games.append({
			"events": [dict(e) for e in events],
			"ids": list(player_ids),
			"winner": winner_id,
		})

	def _event_lines_cost(self, game: dict) -> int:
		# "past" line + translated lines + "end" line; translation does not depend on labels
		probe = _HistoryLabels({})
		count = sum(
			translate_event(e, probe, self.me_id) is not None for e in game["events"]
		)
		return count + 2

	def render(self, current_ids: list[str]) -> list[str]:
		current = labels_for_ids(current_ids, self.me_id)
		others = set(current_ids) - {self.me_id}
		chosen, used = [], 0
		for game in reversed(self.games):
			if not (set(game["ids"]) - {self.me_id}) & others:
				continue
			cost = self._event_lines_cost(game)
			if len(chosen) >= self.max_games or used + cost > self.max_lines:
				break
			chosen.append(game)
			used += cost
		labels = _HistoryLabels(current)
		out: list[str] = []
		for game in reversed(chosen):
			out.append(f"past n {len(game['ids'])}")
			for event in game["events"]:
				line = translate_event(event, labels, self.me_id)
				if line is not None:
					out.append(line)
			winner = game["winner"]
			out.append("end draw" if winner is None else f"end {labels[winner]}")
		return out


def action_candidates(game, me) -> list[Candidate]:
	"""Legal turn actions for player `me` (a Player or a player id)."""
	me_id = me if isinstance(me, str) else me.id
	player = game._get_player(me_id)
	labels = make_labels(game, me_id)
	opponents = [p for p in game.alive_players if p.id != me_id]
	out: list[Candidate] = []
	if player.coins >= 10:
		for opp in opponents:
			out.append(Candidate(f"P0 coup {labels[opp.id]}", ("coup", opp.id), False))
		return out
	out.append(Candidate("P0 inc", ("income", None), False))
	out.append(Candidate("P0 fa", ("foreign_aid", None), False))
	out.append(Candidate("P0 tax", ("tax", None), False))
	out.append(Candidate("P0 xch", ("exchange", None), False))
	for opp in opponents:
		out.append(Candidate(f"P0 steal {labels[opp.id]}", ("steal", opp.id), False))
	if player.coins >= 3:
		for opp in opponents:
			out.append(Candidate(f"P0 kill {labels[opp.id]}", ("assassinate", opp.id), False))
	if player.coins >= 7:
		for opp in opponents:
			out.append(Candidate(f"P0 coup {labels[opp.id]}", ("coup", opp.id), False))
	return out


def decision_candidates(kind: str, context: dict, me_id: str, labels: dict[str, str]) -> list[Candidate]:
	if kind == "challenge":
		return [Candidate("P0 chal", True, False), Candidate("P0 ok", False, False)]
	if kind == "block":
		if me_id not in context["eligible_blockers"]:
			return []
		out = []
		if context["action"] == "foreign_aid":
			out.append(Candidate("P0 blk duke", {"blocker_id": me_id}, False))
		else:
			for role in context["allowed_roles"]:
				out.append(Candidate(
					f"P0 blk {ROLE_ABBR[role]}", {"blocker_id": me_id, "role": role}, False
				))
		out.append(Candidate("P0 ok", None, True))
		return out
	if kind == "reveal_influence":
		out, seen = [], set()
		for card in context["cards"]:
			if card["role"] in seen:
				continue
			seen.add(card["role"])
			out.append(Candidate(f"P0 lose {ROLE_ABBR[card['role']]}", card["index"], False))
		return out
	if kind == "exchange":
		cards = context["cards"]
		out, seen = [], set()
		for combo in itertools.combinations(range(len(cards)), context["keep_count"]):
			roles = tuple(_canon(ROLE_ABBR[cards[i]["role"]] for i in combo))
			if roles in seen:
				continue
			seen.add(roles)
			out.append(Candidate("keep " + " ".join(roles), list(combo), True))
		return out
	raise ValueError(f"Unknown decision type: {kind}")


class Transcript:
	def __init__(self, me_id: str, history: History | None = None):
		self.me_id = me_id
		self.history = history
		self.game = None
		self.labels: dict[str, str] = {}
		self.body: list[str] = []
		self._event_pos = 0
		self._last_state: str | None = None
		self._history_lines: list[str] = []

	# -- setup -----------------------------------------------------------
	def attach(self, game) -> None:
		"""Call after cards are dealt (e.g. from Player.on_game_start)."""
		self.game = game
		self.labels = make_labels(game, self.me_id)
		mine = game._get_player(self.me_id)
		self.body = ["deal " + " ".join(_roles_of(mine.cards))]
		self._last_state = None
		self._event_pos = len(game.events)
		if self.history is not None:
			self._history_lines = self.history.render([p.id for p in game.players])
		else:
			self._history_lines = []

	def history_lines(self) -> list[str]:
		"""Rendered past games (fixed at attach time); empty without history."""
		return list(self._history_lines)

	# -- translation -----------------------------------------------------
	def sync(self) -> None:
		events = self.game.events
		for event in events[self._event_pos:]:
			line = translate_event(event, self.labels, self.me_id)
			if line is not None:
				self.body.append(line)
		self._event_pos = len(events)

	def note(self, line: str) -> None:
		self.body.append(line)

	def _ids_in_label_order(self) -> list[str]:
		return sorted(self.labels, key=lambda pid: int(self.labels[pid][1:]))

	def _state_line(self) -> str:
		coins = " ".join(str(self.game._get_player(pid).coins) for pid in self._ids_in_label_order())
		roles = _roles_of(self.game._get_player(self.me_id).cards)
		return f"$ {coins} |" + "".join(" " + r for r in roles)

	def before_decision(self, kind: str, context: dict | None) -> None:
		self.sync()
		state = self._state_line()
		if state != self._last_state:
			self.note(state)
			self._last_state = state
		if kind == "exchange":
			self.note("draw " + " ".join(ROLE_ABBR[c["role"]] for c in context["cards"]))

	def after_decision(self, candidate: Candidate) -> None:
		if candidate.synthetic:
			self.note(candidate.line)

	# -- candidates ------------------------------------------------------
	def candidates(self, kind: str, context: dict | None) -> list[Candidate]:
		if kind == "action":
			return action_candidates(self.game, self.me_id)
		return decision_candidates(kind, context, self.me_id, self.labels)

	def match(self, kind: str, context: dict | None, answer) -> Candidate | None:
		"""The candidate corresponding to a teacher's answer (None if there is none)."""
		cands = self.candidates(kind, context)
		if kind == "action":
			action, target = answer
			key = (normalize_action(action), target)
			return next((c for c in cands if c.answer == key), None)
		if kind == "challenge":
			return next((c for c in cands if c.answer is bool(answer)), None)
		if kind == "block":
			if answer is None:
				return next((c for c in cands if c.answer is None), None)
			if answer.get("blocker_id") != self.me_id:
				return None
			role = "duke" if context["action"] == "foreign_aid" else answer.get("role")
			return next(
				(c for c in cands if isinstance(c.answer, dict) and c.answer.get("role", "duke") == role),
				None,
			)
		if kind == "reveal_influence":
			role = next((c["role"] for c in context["cards"] if c["index"] == answer), None)
			if role is None:
				return None
			return next((c for c in cands if c.line == f"P0 lose {ROLE_ABBR[role]}"), None)
		if kind == "exchange":
			line = "keep " + " ".join(_canon(ROLE_ABBR[context["cards"][i]["role"]] for i in answer))
			return next((c for c in cands if c.line == line), None)
		raise ValueError(f"Unknown decision type: {kind}")

	# -- end of game -----------------------------------------------------
	def recap(self, winner_id: str) -> None:
		self.note(f"end {self.labels[winner_id]}")
		for pid in self._ids_in_label_order():
			cards = self.game._get_player(pid).cards
			self.note(f"{self.labels[pid]} had " + " ".join(_roles_of(cards, include_revealed=True)))

	def prompt_lines(self, goal: str = "win") -> list[str]:
		"""RULES + history + `now` + `n .. res ..` + body."""
		return (
			rules_lines()
			+ self._history_lines
			+ ["now", result_line(len(self.labels), goal)]
			+ self.body
		)


def record(player, game, history: History | None = None) -> Transcript:
	"""Patch a teacher bot (instance level) so it records a transcript of its game.

	Call after the Game is constructed (cards dealt). Teacher behaviour is unchanged.
	A block decision for which the player is not eligible is not recorded.
	"""
	transcript = Transcript(player.id, history)
	transcript.attach(game)
	orig_choose = player.choose_action
	orig_decide = player.decide
	orig_end = player.on_game_end

	def choose_action(g):
		transcript.before_decision("action", None)
		captured = []
		real = g.perform_action

		def capture(player_id, action, target_id=None, decision_provider=None):
			if not captured:
				# match now: the candidates depend on the pre-action state
				captured.append(transcript.match("action", None, (action, target_id)))
			return real(player_id, action, target_id, decision_provider)

		g.perform_action = capture
		try:
			result = orig_choose(g)
		finally:
			del g.perform_action
		if captured and captured[0] is not None:
			transcript.after_decision(captured[0])
		return result

	def decide(kind, context):
		if kind == "block" and not transcript.candidates(kind, context):
			return orig_decide(kind, context)
		transcript.before_decision(kind, context)
		answer = orig_decide(kind, context)
		cand = transcript.match(kind, context, answer)
		if cand is not None:
			transcript.after_decision(cand)
		return answer

	def on_game_end(winner_id):
		orig_end(winner_id)
		transcript.sync()
		transcript.recap(winner_id)

	player.choose_action = choose_action
	player.decide = decide
	player.on_game_end = on_game_end
	return transcript
