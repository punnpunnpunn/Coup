"""League self-play for lm/rl.py (opt-in via --league).

The learner keeps facing a mix of opponents so it cannot just exploit its current self:
  * scripted bots,
  * the frozen behaviour-cloned reference (always in the pool, never evicted),
  * frozen snapshots of the learner (bounded, spread over training history),
  * "self": the live policy weights (no grad, decisions not recorded).
League members are sampled with prioritised fictitious self-play (PFSP): the harder a
member is for the learner (EMA of the learner's win rate in games it appeared in), the
more often it is picked.

This module needs no torch at import time (only `snapshot_lm` / `load_snapshot_lm`).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from agents.llm_agent import LLMAgent
from lm.generate_data import make_player
from lm.rl import SessionSource

BOTS_ONLY, WITH_BC, WITH_SNAP, WITH_SELF = "bots_only", "with_bc", "with_snap", "with_self"
TYPE_KEYS = (BOTS_ONLY, WITH_BC, WITH_SNAP, WITH_SELF)


# ----------------------------------------------------------------------------
# Pure policy pieces
# ----------------------------------------------------------------------------
def pick_eviction(updates: list[int]) -> int | None:
	"""Index (into `updates`, the snapshot update numbers in ascending order) of the snapshot
	to drop when over capacity: never the oldest or newest; among the others the one whose
	neighbours are closest together (so the pool stays spread over training history; ties go
	to the older one). With fewer than 3 snapshots the oldest goes."""
	if len(updates) < 2:
		return None
	if len(updates) < 3:
		return 0
	best, best_gap = None, None
	for i in range(1, len(updates) - 1):
		gap = updates[i + 1] - updates[i - 1]
		if best_gap is None or gap < best_gap:
			best, best_gap = i, gap
	return best


def pfsp_weight(winrate: float, power: float) -> float:
	return (1.0 - winrate) ** power + 0.05


@dataclass
class Member:
	id: str  # "bc" or "u0010"
	kind: str  # "bc" | "snap"
	update: int
	path: str | None
	scorer: object = None  # what an opponent LLMAgent scores with (None once evicted)
	lm: object = None
	winrate: float = 0.3
	games: int = 0

	def stats(self) -> dict:
		return {"id": self.id, "kind": self.kind, "update": self.update, "path": self.path,
			"winrate": self.winrate, "games": self.games}


class League:
	def __init__(self, max_snapshots: int = 5, decay: float = 0.1, power: float = 2.0,
		init_winrate: float = 0.3, make_scorer=None):
		self.max_snapshots, self.decay, self.power = max_snapshots, decay, power
		self.init_winrate = init_winrate
		self.make_scorer = make_scorer or (lambda lm: lm)
		self.members: list[Member] = []  # in-memory pool: bc first, then snapshots by update
		self.evicted: list[dict] = []
		self.type_stats = {k: [0, 0] for k in TYPE_KEYS}  # cumulative [wins, games]

	# -- membership --------------------------------------------------------
	@property
	def bc(self) -> Member | None:
		return next((m for m in self.members if m.kind == "bc"), None)

	@property
	def snapshots(self) -> list[Member]:
		return sorted((m for m in self.members if m.kind == "snap"), key=lambda m: m.update)

	def add_bc(self, lm) -> Member:
		m = Member("bc", "bc", 0, None, self.make_scorer(lm), lm, self.init_winrate)
		self.members.insert(0, m)
		return m

	def add_snapshot(self, update: int, lm, path: str | None, winrate=None, games=0) -> list[Member]:
		"""Add a frozen snapshot; returns the members evicted to stay within capacity."""
		m = Member(f"u{update:04d}", "snap", update, path, self.make_scorer(lm), lm,
			self.init_winrate if winrate is None else winrate, games)
		self.members.append(m)
		dropped = []
		while len(self.snapshots) > max(self.max_snapshots, 1):
			snaps = self.snapshots
			victim = snaps[pick_eviction([s.update for s in snaps])]
			self.members.remove(victim)
			victim.scorer = victim.lm = None  # evicted from memory; stays on disk
			self.evicted.append(victim.stats())
			dropped.append(victim)
		return dropped

	# -- PFSP ----------------------------------------------------------------
	def weights(self) -> list[float]:
		return [pfsp_weight(m.winrate, self.power) for m in self.members]

	def sample(self, rng: random.Random) -> Member:
		return rng.choices(self.members, weights=self.weights())[0]

	def record_game(self, info: dict | None, won: bool) -> None:
		"""info: {"types": set of {"bc","snap","self"}, "members": [Member...]} of the opponents."""
		if info is None:
			return
		w = 1.0 if won else 0.0
		for m in {id(m): m for m in info["members"]}.values():
			m.winrate += self.decay * (w - m.winrate)
			m.games += 1
		types = info["types"]
		keys = ([BOTS_ONLY] if not types else
			[k for t, k in (("bc", WITH_BC), ("snap", WITH_SNAP), ("self", WITH_SELF)) if t in types])
		for k in keys:
			self.type_stats[k][0] += int(won)
			self.type_stats[k][1] += 1

	# -- persistence ---------------------------------------------------------
	def state_dict(self) -> dict:
		return {"version": 1, "max_snapshots": self.max_snapshots, "decay": self.decay,
			"power": self.power, "members": [m.stats() for m in self.members],
			"evicted": list(self.evicted), "type_stats": {k: list(v) for k, v in self.type_stats.items()},
			"pfsp_weights": {m.id: w for m, w in zip(self.members, self.weights())}}

	def load_state_dict(self, state: dict, loader) -> None:
		"""Restore PFSP stats and re-load the snapshot members: loader(path, update) -> lm.
		The BC member must already be added (its stats are restored). Missing snapshot dirs
		are skipped with a warning."""
		self.evicted = list(state.get("evicted", []))
		for k, v in state.get("type_stats", {}).items():
			if k in self.type_stats:
				self.type_stats[k] = list(v)
		for s in state.get("members", []):
			if s["kind"] == "bc":
				if self.bc is not None:
					self.bc.winrate, self.bc.games = s["winrate"], s["games"]
				continue
			try:
				lm = loader(s["path"], s["update"])
			except Exception as error:  # noqa: BLE001 - keep going without this member
				print(f"[league] cannot restore snapshot {s['id']} from {s['path']}: {error}")
				self.evicted.append(s)
				continue
			self.add_snapshot(s["update"], lm, s["path"], s["winrate"], s["games"])

	def save_json(self, path: str, extra: dict | None = None) -> None:
		tmp = path + ".tmp"
		with open(tmp, "w") as f:
			json.dump({**self.state_dict(), **(extra or {})}, f, indent=1)
		os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Session source with LLM opponents
# ----------------------------------------------------------------------------
class LeagueSessionSource(SessionSource):
	"""Like SessionSource, but each opponent identity is a scripted bot (prob p_bots), the live
	policy (p_self of the rest) or a league member drawn with PFSP. LLM opponents are
	persistent LLMAgents per identity within the session (own history, own KV session);
	their on_decision stays None, so only the learner's decisions are ever recorded."""

	def __init__(self, rng, make_learner, league: League, self_scorer, p_bots=0.4, p_self=0.2,
		history_max_lines=200, **kw):
		super().__init__(rng, make_learner, **kw)
		self.league, self.self_scorer = league, self_scorer
		self.p_bots, self.p_self = p_bots, p_self
		self.history_max_lines = history_max_lines
		self.game_info = None
		self._llms: list[LLMAgent] = []

	def _new_session(self):
		rng = self.rng
		self.pool = []  # entries: (kind, payload, name, member)
		for i in range(rng.randint(self.min_pool, self.max_pool)):
			if rng.random() < self.p_bots or not self.league.members:
				cls = rng.choice(self.teachers)
				self.pool.append(("bot", cls, f"{cls.__name__} {i}", None))
				continue
			if rng.random() < self.p_self:
				kind, member, scorer, name = "self", None, self.self_scorer, f"Self {i}"
			else:
				member = self.league.sample(rng)
				kind, scorer, name = member.kind, member.scorer, f"{member.id} {i}"
			agent = LLMAgent(name, name=name, seed=rng.randrange(2**31), scorer=scorer,
				temperature=1.0, history_max_lines=self.history_max_lines)
			self.pool.append((kind, agent, name, member))
		self.learner = self.make_learner()
		self._left = self.games_per_session

	def next_game(self):
		if self._left <= 0:
			self._new_session()
		self._left -= 1
		rng = self.rng
		k = rng.randint(self.min_players, min(self.max_players, len(self.pool) + 1))
		chosen = rng.sample(self.pool, k - 1)
		players, types, members, self._llms = [], set(), [], []
		for kind, payload, name, member in chosen:
			if kind == "bot":
				players.append(make_player(payload, name, name, rng.randrange(2**31)))
				continue
			players.append(payload)
			self._llms.append(payload)
			types.add("snap" if kind == "snap" else kind)
			if member is not None:
				members.append(member)
		self.game_info = {"types": types, "members": members}
		players.append(self.learner)
		rng.shuffle(players)
		random.seed(rng.randrange(2**31))  # the engine shuffles with the global RNG
		return self.learner, players

	def after_game(self):
		"""Free the opponents' KV caches (the learner's stays, as in the plain source)."""
		for agent in self._llms:
			if agent._session is not None:
				agent._session.reset()


# ----------------------------------------------------------------------------
# Torch helpers
# ----------------------------------------------------------------------------
def snapshot_lm(policy):
	"""Frozen bf16 eval-mode CoupLM holding a copy of the policy's current weights (no
	tensor shared with the policy; the tokenizer is shared, it is immutable)."""
	import torch
	from transformers import AutoModelForCausalLM
	from lm.model import CoupLM

	device = policy.device
	with torch.device(device):
		model = AutoModelForCausalLM.from_config(policy.model.config, dtype=torch.bfloat16)
	with torch.no_grad():
		model.load_state_dict(policy.model.state_dict())  # copies (and casts) every tensor
	model.requires_grad_(False)
	return CoupLM(model, policy.tokenizer, device, max_len=policy.max_len)


def load_snapshot_lm(path, device):
	import torch
	from transformers import AutoModelForCausalLM, AutoTokenizer
	from lm.model import CoupLM

	model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
	model.requires_grad_(False)
	return CoupLM(model, AutoTokenizer.from_pretrained(path), device)
