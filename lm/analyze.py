"""Describe HOW each contender plays (not just how often it wins).

Uses the same schedule and contender specification as lm.compare: every
contender plays the same seat in the same scheduled games. Only the contender
seat is instrumented (instance-level wrappers, behaviour unchanged). Each
decision is recorded together with the private state at decision time, then
turned into behavioural metrics: action mix, bluffing, challenging, blocking,
targeting, reveals/exchanges, outcomes, plus readable example transcripts.

Usage:
	python -m lm.analyze --contenders bc=checkpoints/coup-lm rl=checkpoints/coup-lm-rl \
		--bots DukeFish,FinalAgent --games 200 --out data/analysis.json \
		--examples-dir data/analysis_examples
"""
from __future__ import annotations

import argparse
import json
import os
import re

from lm import compare
from lm.encoding import action_candidates, normalize_action

ACTIONS = ["income", "foreign_aid", "tax", "exchange", "steal", "assassinate", "coup"]
CLAIM_ROLE = {"tax": "duke", "steal": "captain", "assassinate": "assassin", "exchange": "ambassador"}
ROLES = ["duke", "assassin", "captain", "ambassador", "contessa"]
ATTACKS = ("coup", "assassinate", "steal")
BUCKETS = ["0-2", "3-6", "7-9", "10+"]
PHASES = ["early", "mid", "endgame"]


def bucket(coins: int) -> str:
	return "0-2" if coins <= 2 else "3-6" if coins <= 6 else "7-9" if coins <= 9 else "10+"


def phase(alive: int, influence: int) -> str:
	if alive <= 2:
		return "endgame"
	return "early" if influence == 2 else "mid"


# ----------------------------------------------------------------------------
# instrumentation
class Recorder:
	"""Observes one player instance across many games without changing behaviour.

	Wraps on_game_start / choose_action / decide on the instance. The previous game is
	finalized when the next one begins (or on flush()), so the caller's loop is untouched.
	"""

	def __init__(self, player):
		self.player = player
		self.games: list[dict] = []
		self.game = None
		self._orig: dict = {}
		self._saved: dict = {}

	# -- install / uninstall ------------------------------------------------
	def install(self):
		p = self.player
		for name in ("on_game_start", "choose_action", "decide"):
			self._orig[name] = getattr(p, name)
			if name in p.__dict__:
				self._saved[name] = p.__dict__[name]
		p.on_game_start = self._on_game_start
		p.choose_action = self._choose_action
		p.decide = self._decide
		return self

	def uninstall(self):
		for name in self._orig:
			self.player.__dict__.pop(name, None)
			if name in self._saved:
				self.player.__dict__[name] = self._saved[name]
		self._orig, self._saved = {}, {}

	def flush(self):
		"""Finalize the game in progress / just finished."""
		if self.game is not None:
			self.games.append(self._finalize())
			self.game = None

	# -- per game state -----------------------------------------------------
	def begin(self, game):
		if game is not self.game:
			self.flush()
		self.game = game
		self.n0 = len(game.players)
		self.initial = {p.id: [c.role.value for c in p.cards] for p in game.players}
		self.classes = {p.id: type(p).__name__ for p in game.players}
		self.decisions: list[dict] = []
		self._pos = 0
		self._actions_seen = 0
		self._last_attacker = None

	def _on_game_start(self, game):
		self._orig["on_game_start"](game)
		self.begin(game)

	def _sync(self):
		events = self.game.events
		me = self.player.id
		for e in events[self._pos:]:
			if e["type"] == "action":
				self._actions_seen += 1
				if e["target"] == me and e["action"] in ATTACKS:
					self._last_attacker = e["actor"]
		self._pos = len(events)

	def _hidden(self):
		return sorted(c.role.value for c in self.player.cards if not c.revealed)

	def _snapshot(self, kind):
		self._sync()
		g, me = self.game, self.player
		opps = [{"id": p.id, "cls": self.classes[p.id], "coins": p.coins, "influence": p.influence}
			for p in g.alive_players if p is not me]
		turn = self._actions_seen + (1 if kind == "action" else 0)
		la = self._last_attacker
		if la is not None and not g._get_player(la).alive:
			la = None
		d = {
			"kind": kind, "event_idx": len(g.events), "turn": turn,
			"round": (max(turn, 1) - 1) // self.n0 + 1, "n_players": self.n0,
			"alive": len(g.alive_players), "coins": me.coins, "influence": me.influence,
			"hidden": self._hidden(), "opponents": opps, "last_attacker": la,
		}
		self.decisions.append(d)
		return d

	@staticmethod
	def _target_info(d, target_id):
		opps = d["opponents"]
		t = next((o for o in opps if o["id"] == target_id), None)
		if t is None:
			return None
		cr = 1 + sum(o["coins"] > t["coins"] for o in opps)
		ir = 1 + sum(o["influence"] > t["influence"] for o in opps)
		return {"id": t["id"], "cls": t["cls"], "coins": t["coins"], "influence": t["influence"],
			"coins_rank": cr, "influence_rank": ir, "coin_leader": cr == 1,
			"influence_leader": ir == 1, "last_attacker": t["id"] == d["last_attacker"]}

	@staticmethod
	def _claim_kind(events):
		"""'block' if the claim being challenged is a block, else the turn action's claim.
		Earlier challengers' passes (no_challenge) sit between the claim and this decision."""
		for e in reversed(events):
			if e["type"] != "no_challenge":
				return "block" if e["type"] == "block" else "action"
		return "action"

	# -- wrappers -----------------------------------------------------------
	def _choose_action(self, g):
		d = self._snapshot("action")
		d["options"] = [[normalize_action(a), t] for (a, t) in
			(c.answer for c in action_candidates(g, self.player))]
		captured = []
		real = g.perform_action

		def capture(player_id, action, target_id=None, decision_provider=None):
			captured[:] = [(normalize_action(action), target_id)]
			return real(player_id, action, target_id, decision_provider)

		g.perform_action = capture
		try:
			result = self._orig["choose_action"](g)
		finally:
			del g.perform_action
		if captured:
			action, target = captured[0]
			d["choice"] = {"action": action, "target": target}
			d["target"] = self._target_info(d, target) if target else None
		else:
			d["choice"] = None
		return result

	def _decide(self, kind, context):
		orig = self._orig["decide"]
		g, me = self.game, self.player
		if kind == "block" and me.id not in context["eligible_blockers"]:
			return orig(kind, context)
		d = self._snapshot(kind)
		if kind == "challenge":
			claimant = g._get_player(context["claimant_id"])
			role = context["role"]
			d.update({
				"role": role, "claimant": claimant.id, "claimant_cls": self.classes[claimant.id],
				"claim_kind": self._claim_kind(g.events),
				"claimant_has_role": any(c.role.value == role and not c.revealed for c in claimant.cards),
				"options": [True, False],
			})
		elif kind == "block":
			roles = list(context["allowed_roles"])
			d.update({"action": context["action"], "options": roles + [None],
				"holds_role": any(r in d["hidden"] for r in roles)})
		elif kind == "reveal_influence":
			d["options"] = [c["role"] for c in context["cards"]]
		elif kind == "exchange":
			d["options"] = [c["role"] for c in context["cards"]]
			d["keep_count"] = context["keep_count"]
		answer = orig(kind, context)
		if kind == "challenge":
			d["choice"] = bool(answer)
		elif kind == "block":
			d["choice"] = None if answer is None else (
				"duke" if context["action"] == "foreign_aid" else answer.get("role"))
		elif kind == "reveal_influence":
			d["choice"] = next((c["role"] for c in context["cards"] if c["index"] == answer), None)
		elif kind == "exchange":
			d["choice"] = [context["cards"][i]["role"] for i in answer]
			d["kept_original"] = all(i < len(d["hidden"]) for i in answer)
		return answer

	# -- finalization -------------------------------------------------------
	def _finalize(self) -> dict:
		g, me = self.game, self.player.id
		self._sync()
		events = g.events
		# derived from events only: opponents persist across games, so live player
		# state (cards, alive) is overwritten when the next game deals
		winner = next((e["player"] for e in events if e["type"] == "win"), None)
		outcome = "win" if winner == me else ("loss" if winner is not None else "draw")
		decs_action = {d["event_idx"]: d for d in self.decisions if d["kind"] == "action"}
		block_decs = [d for d in self.decisions if d["kind"] == "block" and d.get("choice")]
		claims = []

		def challenge_result(i, role):
			j = i + 1
			while j < len(events) and events[j]["type"] == "no_challenge":
				j += 1
			if (j < len(events) and events[j]["type"] == "challenge"
					and events[j]["claimant"] == me and events[j]["role"] == role):
				return True, j + 1 < len(events) and events[j + 1]["type"] == "claim_failed"
			return False, False

		for i, e in enumerate(events):
			if e["type"] == "action" and e["actor"] == me and e["action"] in CLAIM_ROLE:
				d = decs_action.get(i)
				role = CLAIM_ROLE[e["action"]]
				hidden = d["hidden"] if d else []
				key, kind = e["action"], "action"
			elif e["type"] == "block" and e["blocker"] == me:
				prior = [d for d in block_decs if d["event_idx"] <= i]
				hidden = prior[-1]["hidden"] if prior else []
				role, key, kind = e["role"], "block:" + e["action"], "block"
			else:
				continue
			challenged, caught = challenge_result(i, role)
			claims.append({"event_idx": i, "role": role, "key": key, "kind": kind,
				"bluff": role not in hidden, "challenged": challenged, "caught": caught})

		reveals = {}
		elim = []
		first_loss = None
		n_actions = 0
		for i, e in enumerate(events):
			if e["type"] == "action":
				n_actions += 1
			elif e["type"] == "reveal":
				pid = e["player"]
				reveals[pid] = reveals.get(pid, 0) + 1
				if reveals[pid] == len(self.initial[pid]):
					elim.append(pid)
				if pid == me and first_loss is None:
					first_loss = {"event_idx": i, "turn": n_actions,
						"round": (max(n_actions, 1) - 1) // self.n0 + 1}
		if outcome == "win":
			place = self.n0
		elif me in elim:
			place = 1 + elim.index(me)
		else:
			place = None
		own_turns = sum(e["type"] == "action" and e["actor"] == me for e in events)
		return {
			"index": len(self.games), "n_players": self.n0, "outcome": outcome,
			"own_turns": own_turns, "total_turns": n_actions, "place": place,
			"first_loss": first_loss, "decisions": self.decisions, "claims": claims,
			"initial": self.initial, "classes": self.classes,
			"log": list(g.log), "events": [dict(e) for e in events],
		}


# ----------------------------------------------------------------------------
# metrics
def ratio(a, b):
	return a / b if b else None


def _dist(counter: dict, keys=None):
	n = sum(counter.values())
	keys = keys or sorted(counter)
	return {"n": n, "counts": {k: counter.get(k, 0) for k in keys},
		"freq": {k: ratio(counter.get(k, 0), n) for k in keys}}


def action_metrics(games):
	overall = {a: 0 for a in ACTIONS}
	by_b = {b: {a: 0 for a in ACTIONS} for b in BUCKETS}
	by_p = {p: {a: 0 for a in ACTIONS} for p in PHASES}
	for g in games:
		for d in g["decisions"]:
			if d["kind"] != "action" or not d.get("choice"):
				continue
			a = d["choice"]["action"]
			overall[a] += 1
			by_b[bucket(d["coins"])][a] += 1
			by_p[phase(d["alive"], d["influence"])][a] += 1
	return {"overall": _dist(overall, ACTIONS),
		"by_coin_bucket": {b: _dist(c, ACTIONS) for b, c in by_b.items()},
		"by_phase": {p: _dist(c, ACTIONS) for p, c in by_p.items()}}


def _claim_stats(claims):
	bl = [c for c in claims if c["bluff"]]
	hon = [c for c in claims if not c["bluff"]]
	bl_ch = sum(c["challenged"] for c in bl)
	return {
		"claims": len(claims), "bluffs": len(bl), "bluff_rate": ratio(len(bl), len(claims)),
		"bluffs_challenged": bl_ch, "bluff_challenged_rate": ratio(bl_ch, len(bl)),
		"bluffs_caught": sum(c["caught"] for c in bl), "caught_given_challenged": ratio(sum(c["caught"] for c in bl), bl_ch),
		"honest_claims": len(hon), "honest_challenged": sum(c["challenged"] for c in hon),
		"honest_challenged_rate": ratio(sum(c["challenged"] for c in hon), len(hon)),
	}


def bluff_metrics(games):
	claims = [c for g in games for c in g["claims"]]
	keys = sorted({c["key"] for c in claims})
	return {
		"all": _claim_stats(claims),
		"by_role": {r: _claim_stats([c for c in claims if c["role"] == r]) for r in ROLES
			if any(c["role"] == r for c in claims)},
		"by_claim": {k: _claim_stats([c for c in claims if c["key"] == k]) for k in keys},
	}


def _chal_stats(ds):
	ch = [d for d in ds if d["choice"]]
	acc = sum(not d["claimant_has_role"] for d in ch)
	bl = [d for d in ds if not d["claimant_has_role"]]
	return {
		"opportunities": len(ds), "challenges": len(ch), "challenge_rate": ratio(len(ch), len(ds)),
		"claimant_bluffing": len(bl), "bluff_base_rate": ratio(len(bl), len(ds)),
		"correct_challenges": acc, "accuracy": ratio(acc, len(ch)),
		"missed_bluffs": sum(not d["choice"] for d in bl),
		"missed_bluff_rate": ratio(sum(not d["choice"] for d in bl), len(bl)),
	}


def challenge_metrics(games):
	ds = [d for g in games for d in g["decisions"] if d["kind"] == "challenge"]
	return {
		"all": _chal_stats(ds),
		"by_role": {r: _chal_stats([d for d in ds if d["role"] == r]) for r in ROLES
			if any(d["role"] == r for d in ds)},
		"by_claim_kind": {k: _chal_stats([d for d in ds if d["claim_kind"] == k])
			for k in ("action", "block") if any(d["claim_kind"] == k for d in ds)},
	}


def _block_stats(ds):
	bl = [d for d in ds if d["choice"] is not None]
	bluff = [d for d in bl if d["choice"] not in d["hidden"]]
	holds = [d for d in ds if d["holds_role"]]
	nots = [d for d in ds if not d["holds_role"]]
	return {
		"eligible": len(ds), "blocks": len(bl), "block_rate": ratio(len(bl), len(ds)),
		"bluff_blocks": len(bluff), "bluff_block_fraction": ratio(len(bluff), len(bl)),
		"block_rate_when_holding_role": ratio(sum(d["choice"] is not None for d in holds), len(holds)),
		"block_rate_when_not_holding": ratio(sum(d["choice"] is not None for d in nots), len(nots)),
	}


def block_metrics(games):
	ds = [d for g in games for d in g["decisions"] if d["kind"] == "block"]
	acts = sorted({d["action"] for d in ds})
	return {"all": _block_stats(ds),
		"by_action": {a: _block_stats([d for d in ds if d["action"] == a]) for a in acts}}


def _target_stats(ds):
	ds = [d for d in ds if d.get("target")]
	cont = [d for d in ds if len(d["opponents"]) >= 2]  # a real choice of target
	out = {"n": len(ds), "n_contested": len(cont)}
	for name, key, flag in (("coin_leader", "coins", "coin_leader"),
			("influence_leader", "influence", "influence_leader")):
		hits = sum(d["target"][flag] for d in cont)
		chance = sum(sum(o[key] == max(p[key] for p in d["opponents"]) for o in d["opponents"])
			/ len(d["opponents"]) for d in cont)
		out[name] = {"rate": ratio(hits, len(cont)), "chance": ratio(chance, len(cont))}
	la = [d for d in cont if d["last_attacker"] is not None]
	out["last_attacker"] = {
		"n_available": len(la), "rate": ratio(sum(d["target"]["last_attacker"] for d in la), len(la)),
		"chance": ratio(sum(1 / len(d["opponents"]) for d in la), len(la))}
	classes = sorted({d["target"]["cls"] for d in ds} | {o["cls"] for d in ds for o in d["opponents"]})
	out["by_bot_type"] = {c: {
		"share": ratio(sum(d["target"]["cls"] == c for d in ds), len(ds)),
		"chance": ratio(sum(sum(o["cls"] == c for o in d["opponents"]) / len(d["opponents"]) for d in ds), len(ds)),
	} for c in classes}
	return out


def target_metrics(games):
	decs = [d for g in games for d in g["decisions"] if d["kind"] == "action" and d.get("choice")]
	out = {"all": _target_stats([d for d in decs if d["choice"]["action"] in ATTACKS])}
	for a in ATTACKS:
		out[a] = _target_stats([d for d in decs if d["choice"]["action"] == a])
	coups = [d for d in decs if d["choice"]["action"] == "coup"]
	forced = [d for d in decs if d["coins"] >= 10]
	hist: dict[str, int] = {}
	for d in coups:
		hist[str(d["coins"])] = hist.get(str(d["coins"]), 0) + 1
	games_forced = sum(any(d["kind"] == "action" and d["coins"] >= 10 for d in g["decisions"]) for g in games)
	out["coup_coin_level"] = {
		"n": len(coups), "mean": ratio(sum(d["coins"] for d in coups), len(coups)),
		"histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
		"voluntary_coups": sum(d["coins"] < 10 for d in coups), "forced_coups": sum(d["coins"] >= 10 for d in coups)}
	out["forced_coup"] = {"decisions": len(forced), "games_reaching_10": games_forced,
		"games_reaching_10_rate": ratio(games_forced, len(games))}
	return out


def reveal_metrics(games):
	rev = [d for g in games for d in g["decisions"] if d["kind"] == "reveal_influence"]
	ch = [d for d in rev if len(set(d["options"])) > 1]  # a genuine choice
	gave = {r: sum(d["choice"] == r for d in ch) for r in ROLES}
	held = {r: sum(r in d["options"] for d in ch) for r in ROLES}
	ex = [d for g in games for d in g["decisions"] if d["kind"] == "exchange"]
	kept = {r: sum(d["choice"].count(r) for d in ex) for r in ROLES}
	offered = {r: sum(d["options"].count(r) for d in ex) for r in ROLES}
	return {
		"reveal": {"decisions": len(rev), "with_choice": len(ch),
			"given_up": gave, "held_in_choice": held,
			"given_up_rate": {r: ratio(gave[r], held[r]) for r in ROLES},
			"given_up_share": {r: ratio(gave[r], len(ch)) for r in ROLES}},
		"exchange": {"n": len(ex), "kept": kept, "offered": offered,
			"keep_rate": {r: ratio(kept[r], offered[r]) for r in ROLES},
			"kept_original_hand_rate": ratio(sum(d["kept_original"] for d in ex), len(ex))},
	}


def outcome_metrics(games, early_turn):
	n = len(games)
	wins = sum(g["outcome"] == "win" for g in games)
	draws = sum(g["outcome"] == "draw" for g in games)
	elim = [g for g in games if g["outcome"] == "loss"]
	places: dict[str, dict[str, int]] = {}
	norm = []
	for g in games:
		if g["place"] is None:
			continue
		p = places.setdefault(str(g["n_players"]), {})
		p[str(g["place"])] = p.get(str(g["place"]), 0) + 1
		norm.append((g["place"] - 1) / (g["n_players"] - 1))
	def split(sel):
		gs = [g for g in games if sel(g)]
		w = sum(g["outcome"] == "win" for g in gs)
		return {"games": len(gs), "wins": w, "win_rate": ratio(w, len(gs))}
	return {
		"games": n, "wins": wins, "losses": len(elim), "draws": draws, "win_rate": ratio(wins, n),
		"avg_turns_taken": ratio(sum(g["own_turns"] for g in games), n),
		"avg_turns_taken_when_eliminated": ratio(sum(g["own_turns"] for g in elim), len(elim)),
		"finish_place_counts_by_players": places,
		"mean_normalized_place": ratio(sum(norm), len(norm)),
		"first_card_loss": {
			"early_round_max": early_turn,
			"early": split(lambda g: g["first_loss"] and g["first_loss"]["round"] <= early_turn),
			"late": split(lambda g: g["first_loss"] and g["first_loss"]["round"] > early_turn),
			"never_lost_a_card": split(lambda g: g["first_loss"] is None),
		},
	}


def compute_metrics(games, early_turn=3):
	return {
		"outcomes": outcome_metrics(games, early_turn),
		"actions": action_metrics(games),
		"bluffing": bluff_metrics(games),
		"challenging": challenge_metrics(games),
		"blocking": block_metrics(games),
		"targeting": target_metrics(games),
		"reveals_exchanges": reveal_metrics(games),
	}


# ----------------------------------------------------------------------------
# example transcripts
def _fmt_option(o):
	return o if not isinstance(o, list) else (o[0] if o[1] is None else f"{o[0]}>{o[1]}")


def describe_decision(d) -> str:
	head = (f">> [{d['kind']}] hidden={d['hidden']} coins={d['coins']} "
		f"opponents={[(o['id'], o['coins'], o['influence']) for o in d['opponents']]}")
	if d["kind"] == "action":
		c = d.get("choice")
		opts = sorted({o[0] for o in d["options"]})
		choice = "?" if not c else c["action"] + (f" > {c['target']}" if c["target"] else "")
		return f"{head}\n   options: {', '.join(opts)}\n   chose: {choice}"
	if d["kind"] == "challenge":
		truth = "claimant HAS it" if d["claimant_has_role"] else "claimant is BLUFFING"
		return f"{head}\n   {d['claimant']} claims {d['role']} ({d['claim_kind']}); truth: {truth}\n   chose: {'CHALLENGE' if d['choice'] else 'pass'}"
	if d["kind"] == "block":
		return f"{head}\n   block {d['action']} with {d['options'][:-1]}\n   chose: {d['choice'] or 'no block'}"
	return f"{head}\n   options: {d['options']}\n   chose: {d['choice']}"


def render_transcript(name: str, g: dict) -> str:
	me_hands = ""
	out = [f"Contender {name}  game {g['index']}  players {g['n_players']}  outcome {g['outcome'].upper()}",
		"Hidden cards of every player are shown for analysis. '>>' marks a contender decision.",
		"Initial hands:"]
	for pid, hand in g["initial"].items():
		mark = " (contender)" if pid == "contender" else ""
		out.append(f"  {pid} [{g['classes'][pid]}]{mark}: {', '.join(hand)}")
	out.append("-" * 70)
	by_idx: dict[int, list] = {}
	for d in g["decisions"]:
		by_idx.setdefault(d["event_idx"], []).append(d)
	claim_at = {c["event_idx"]: c for c in g["claims"]}
	for i, line in enumerate(g["log"]):
		for d in by_idx.get(i, []):
			out.append(describe_decision(d))
		if g["events"][i]["type"] == "turn":
			out.append("")
		tag = ""
		c = claim_at.get(i)
		if c:
			tag = f"   <-- contender {'BLUFF' if c['bluff'] else 'honest'} {c['role']} claim"
			if c["challenged"]:
				tag += ", challenged" + (" and CAUGHT" if c["caught"] else ", survived")
		out.append(line + tag)
	for d in by_idx.get(len(g["log"]), []):
		out.append(describe_decision(d))
	return "\n".join(out) + "\n"


def pick_examples(games):
	"""[(label, game)]: a win, a loss, and the game with the most bluffs (distinct if possible)."""
	chosen: list[tuple[str, dict]] = []
	used = set()

	def add(label, g):
		if g is not None and g["index"] not in used:
			used.add(g["index"])
			chosen.append((label, g))

	bluffs = lambda g: sum(c["bluff"] for c in g["claims"])
	add("win", next((g for g in games if g["outcome"] == "win"), None))
	add("loss", next((g for g in games if g["outcome"] == "loss"), None))
	rest = sorted((g for g in games if g["index"] not in used), key=lambda g: (-bluffs(g), g["index"]))
	add("most_bluffs", rest[0] if rest else None)
	# if win/loss was missing, top up to three with the next bluffiest games
	for g in rest[1:]:
		if len(chosen) >= 3:
			break
		add("extra", g)
	return chosen


def safe_name(s: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def save_examples(name: str, games, directory: str) -> list[str]:
	os.makedirs(directory, exist_ok=True)
	paths = []
	for label, g in pick_examples(games):
		path = os.path.join(directory, f"{safe_name(name)}_{label}_game{g['index']}.txt")
		with open(path, "w", encoding="utf-8") as f:
			f.write(render_transcript(name, g))
		paths.append(path)
	return paths


# ----------------------------------------------------------------------------
# running
def analyze(contenders, seed, games, counts, pool_size, early_turn=3, examples_dir=None,
		log=print, progress_every: int = 25) -> dict:
	spec, slots = compare.make_schedule(seed, games, counts, pool_size)
	result = {}
	for name, factory in contenders:
		recorders = []

		def instrumented(factory=factory):
			player = factory()
			recorders.append(Recorder(player).install())
			return player

		records = compare.run_contender(name, instrumented, spec, slots, progress_every, log)
		rec = recorders[0]
		rec.flush()
		recorded = [g["outcome"] for g in rec.games]
		if recorded != [r["outcome"] for r in records]:
			raise RuntimeError(f"recorder outcomes disagree with compare for {name}")
		entry = compute_metrics(rec.games, early_turn)
		if examples_dir:
			entry["examples"] = save_examples(name, rec.games, examples_dir)
		result[name] = entry
	return {"seed": seed, "games": games, "players": counts, "pool_size": pool_size,
		"early_round_max": early_turn, "contenders": result}


# ----------------------------------------------------------------------------
# text summary
def pc(x):
	return "   -  " if x is None else f"{100 * x:5.1f}%"


def num(x, f="{:.2f}"):
	return "  -  " if x is None else f.format(x)


def print_summary(result: dict) -> None:
	cs = result["contenders"]
	names = list(cs)
	w = max(10, *(len(n) + 1 for n in names))
	label_w = 40

	def row(label, fn):
		vals = []
		for n in names:
			try:
				vals.append(fn(cs[n]))
			except (KeyError, TypeError):
				vals.append("-")
		print(f"{label:<{label_w}}" + "".join(f"{v:>{w}}" for v in vals))

	def section(title):
		print(f"\n== {title} " + "=" * max(0, label_w + w * len(names) - len(title) - 4))
		print(f"{'':<{label_w}}" + "".join(f"{n:>{w}}" for n in names))

	print(f"Games: {result['games']}  seed: {result['seed']}  players: {result['players']}")
	section("Outcomes")
	row("win rate", lambda m: pc(m["outcomes"]["win_rate"]))
	row("wins / draws", lambda m: f"{m['outcomes']['wins']}/{m['outcomes']['draws']}")
	row("avg turns taken", lambda m: num(m["outcomes"]["avg_turns_taken"], "{:.1f}"))
	row("mean place (0=first out, 1=winner)", lambda m: num(m["outcomes"]["mean_normalized_place"]))
	row("win rate | lost 1st card early", lambda m: pc(m["outcomes"]["first_card_loss"]["early"]["win_rate"]))
	row("win rate | lost 1st card late", lambda m: pc(m["outcomes"]["first_card_loss"]["late"]["win_rate"]))
	section("Actions (overall)")
	for a in ACTIONS:
		row(a, lambda m, a=a: pc(m["actions"]["overall"]["freq"][a]))
	section("Actions by phase: tax / steal+assassinate+coup share")
	for p in PHASES:
		row(f"{p} tax", lambda m, p=p: pc(m["actions"]["by_phase"][p]["freq"]["tax"]))
		row(f"{p} attack", lambda m, p=p: pc(sum(m["actions"]["by_phase"][p]["freq"][a] or 0 for a in ATTACKS)
			if m["actions"]["by_phase"][p]["n"] else None))
	section("Actions by coin bucket: income / attack share")
	for b in BUCKETS:
		row(f"{b} income", lambda m, b=b: pc(m["actions"]["by_coin_bucket"][b]["freq"]["income"]))
		row(f"{b} attack", lambda m, b=b: pc(sum(m["actions"]["by_coin_bucket"][b]["freq"][a] or 0 for a in ATTACKS)
			if m["actions"]["by_coin_bucket"][b]["n"] else None))
	section("Bluffing (own claims made without the role)")
	row("all claims: bluff rate", lambda m: pc(m["bluffing"]["all"]["bluff_rate"]))
	for r in ROLES:
		row(f"  {r} bluff rate", lambda m, r=r: pc(m["bluffing"]["by_role"][r]["bluff_rate"]))
	row("bluffs challenged", lambda m: pc(m["bluffing"]["all"]["bluff_challenged_rate"]))
	row("challenged bluffs caught", lambda m: pc(m["bluffing"]["all"]["caught_given_challenged"]))
	row("honest claims challenged", lambda m: pc(m["bluffing"]["all"]["honest_challenged_rate"]))
	section("Challenging")
	row("challenge rate (when able)", lambda m: pc(m["challenging"]["all"]["challenge_rate"]))
	row("accuracy (claimant bluffing)", lambda m: pc(m["challenging"]["all"]["accuracy"]))
	row("base rate of bluffs faced", lambda m: pc(m["challenging"]["all"]["bluff_base_rate"]))
	row("missed-bluff rate", lambda m: pc(m["challenging"]["all"]["missed_bluff_rate"]))
	for r in ROLES:
		row(f"  challenge rate vs {r}", lambda m, r=r: pc(m["challenging"]["by_role"][r]["challenge_rate"]))
	section("Blocking")
	row("block rate (when eligible)", lambda m: pc(m["blocking"]["all"]["block_rate"]))
	for a in ("foreign_aid", "assassinate", "steal"):
		row(f"  {a}", lambda m, a=a: pc(m["blocking"]["by_action"][a]["block_rate"]))
	row("bluff-block fraction", lambda m: pc(m["blocking"]["all"]["bluff_block_fraction"]))
	section("Targeting (rate; chance in brackets)")
	for k, lab in (("coin_leader", "coin leader"), ("influence_leader", "influence leader")):
		row(f"{lab} (any attack)", lambda m, k=k: f"{pc(m['targeting']['all'][k]['rate']).strip()}"
			f"({100 * m['targeting']['all'][k]['chance']:.0f})")
	row("last attacker", lambda m: f"{pc(m['targeting']['all']['last_attacker']['rate']).strip()}"
		f"({100 * m['targeting']['all']['last_attacker']['chance']:.0f})")
	row("coup at avg coins", lambda m: num(m["targeting"]["coup_coin_level"]["mean"], "{:.1f}"))
	row("forced coup (games reaching 10)", lambda m: pc(m["targeting"]["forced_coup"]["games_reaching_10_rate"]))
	section("Reveals and exchanges")
	for r in ROLES:
		row(f"give up {r} (when held in choice)", lambda m, r=r: pc(m["reveals_exchanges"]["reveal"]["given_up_rate"][r]))
	for r in ROLES:
		row(f"exchange keeps {r}", lambda m, r=r: pc(m["reveals_exchanges"]["exchange"]["keep_rate"][r]))
	row("exchange keeps original hand", lambda m: pc(m["reveals_exchanges"]["exchange"]["kept_original_hand_rate"]))


# ----------------------------------------------------------------------------
def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--contenders", nargs="*", default=[], metavar="NAME=PATH",
		help="LM contenders, each name=checkpoint_path")
	p.add_argument("--bots", default=None, help="'all' or a comma list of teacher bot class names")
	p.add_argument("--games", type=int, default=200)
	p.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6])
	p.add_argument("--pool-size", type=int, default=6)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--greedy", action="store_true")
	p.add_argument("--temperature", type=float, default=1.0)
	p.add_argument("--early-turn", type=int, default=3,
		help="first card lost in round <= this counts as early (round = turns / players)")
	p.add_argument("--out", default="data/analysis.json")
	p.add_argument("--examples-dir", default="data/analysis_examples")
	args = p.parse_args(argv)
	if any(not 2 <= k <= 6 for k in args.players):
		p.error("player counts must be in 2..6")

	contenders = compare.build_contenders(args, p)
	result = analyze(contenders, args.seed, args.games, args.players, args.pool_size,
		args.early_turn, args.examples_dir)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=1)
	print_summary(result)
	print(f"\nmetrics -> {args.out}\nexamples -> {args.examples_dir}")


if __name__ == "__main__":
	main()
