"""Same-seat comparison of LM checkpoints and scripted bots.

Every contender plays the same seat in the same scheduled games (common random
numbers for the setup), so differences in win rate are paired comparisons.

Usage:
	python -m lm.compare --contenders untrained=checkpoints/coup-lm-untrained \
		bc=checkpoints/coup-lm rl=checkpoints/coup-lm-rl --bots all --games 300 \
		--plot data/compare.png
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass

from agents.llm_agent import LLMAgent
from lm.generate_data import MAX_TURNS_PER_PLAYER, TEACHERS, make_player, play_game
from src.game import Game

CONTENDER_ID = "contender"


# ----------------------------------------------------------------------------
# statistics
def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
	"""Wilson score interval for a binomial proportion."""
	if n == 0:
		return (0.0, 1.0)
	p = wins / n
	denom = 1 + z * z / n
	centre = (p + z * z / (2 * n)) / denom
	half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
	return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test_p(b: int, c: int) -> float:
	"""Exact two-sided sign (McNemar) test on discordant counts b and c."""
	n = b + c
	if n == 0:
		return 1.0
	k = min(b, c)
	tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
	return min(1.0, 2 * tail)


# ----------------------------------------------------------------------------
# schedule
@dataclass
class Slot:
	index: int
	n_players: int
	opponents: list[int]  # pool indices of the opponents at the table
	seats: list[int]  # seat order; -1 is the contender, otherwise a pool index
	game_seed: int  # global random seed (deck shuffle, engine randomness)
	opp_seeds: dict[int, int]  # pool index -> rng seed for this game
	contender_seed: int  # rng seed for the contender for this game

	@property
	def contender_seat(self) -> int:
		return self.seats.index(-1)


def pool_spec(rng: random.Random, pool_size: int):
	"""[(class, player_id, init_seed)] persistent opponent identities."""
	spec = []
	for i in range(pool_size):
		cls = rng.choice(TEACHERS)
		spec.append((cls, f"{cls.__name__} {i}", rng.randrange(2**31)))
	return spec


def make_schedule(seed: int, games: int, counts: list[int], pool_size: int):
	rng = random.Random(seed)
	spec = pool_spec(rng, pool_size)
	slots = []
	for i in range(games):
		k = min(rng.choice(counts), pool_size + 1)
		opponents = rng.sample(range(pool_size), k - 1)
		seats = [-1] + opponents
		rng.shuffle(seats)
		slots.append(Slot(
			index=i, n_players=k, opponents=opponents, seats=seats,
			game_seed=rng.randrange(2**31),
			opp_seeds={o: rng.randrange(2**31) for o in opponents},
			contender_seed=rng.randrange(2**31),
		))
	return spec, slots


# ----------------------------------------------------------------------------
# running
def _reseed(player, seed: int) -> None:
	rng = getattr(player, "rng", None)
	if isinstance(rng, random.Random):
		rng.seed(seed)


def play_slot(slot: Slot, contender, opponents: dict[int, object], spec):
	"""Play one scheduled game. Returns (outcome, hands, opponent class names)."""
	players = [contender if s == -1 else opponents[s] for s in slot.seats]
	_reseed(contender, slot.contender_seed)
	for o in slot.opponents:
		_reseed(opponents[o], slot.opp_seeds[o])
	random.seed(slot.game_seed)  # the engine shuffles with the global RNG
	with contextlib.redirect_stdout(io.StringIO()):
		game = Game(players)
		hands = [[c.role.value for c in p.cards] for p in players]
		draw = play_game(game)
	if draw:
		outcome = "draw"
	else:
		outcome = "win" if game.winner is contender else "loss"
	return outcome, hands


def run_contender(name: str, factory, spec, slots, progress_every: int = 25, log=print):
	"""Play every slot with one contender instance and persistent opponents."""
	opponents = {
		i: make_player(cls, pid, pid, seed) for i, (cls, pid, seed) in enumerate(spec)
	}
	contender = factory()
	records = []
	t0 = time.time()
	for slot in slots:
		outcome, hands = play_slot(slot, contender, opponents, spec)
		records.append({"outcome": outcome, "hands": hands})
		if progress_every and (slot.index + 1) % progress_every == 0:
			log(f"[{name}] {slot.index + 1}/{len(slots)} games, {time.time() - t0:.0f}s")
	return records


# ----------------------------------------------------------------------------
# reporting
def summarize(records, spec, slots) -> dict:
	n = len(records)
	wins = sum(r["outcome"] == "win" for r in records)
	draws = sum(r["outcome"] == "draw" for r in records)
	lo, hi = wilson(wins, n)
	by_n: dict[int, list[int]] = {}
	by_bot: dict[str, list[int]] = {}
	for slot, r in zip(slots, records):
		won = r["outcome"] == "win"
		st = by_n.setdefault(slot.n_players, [0, 0])
		st[0] += won
		st[1] += 1
		for cname in {spec[o][0].__name__ for o in slot.opponents}:
			sb = by_bot.setdefault(cname, [0, 0])
			sb[0] += won
			sb[1] += 1

	def pack(d):
		out = {}
		for key in sorted(d):
			w, g = d[key]
			l, h = wilson(w, g)
			out[str(key)] = {"wins": w, "games": g, "win_rate": w / g, "ci95": [l, h]}
		return out

	return {
		"games": n, "wins": wins, "draws": draws,
		"win_rate": wins / n if n else 0.0, "ci95": [lo, hi],
		"by_players": pack(by_n), "by_opponent_bot": pack(by_bot),
		"outcomes": [r["outcome"] for r in records],
	}


def paired(ref: dict, other: dict) -> dict:
	a = [o == "win" for o in ref["outcomes"]]
	b = [o == "win" for o in other["outcomes"]]
	ref_only = [i for i, (x, y) in enumerate(zip(a, b)) if x and not y]
	other_only = [i for i, (x, y) in enumerate(zip(a, b)) if y and not x]
	n = len(a)
	return {
		"reference_only_wins": len(ref_only), "other_only_wins": len(other_only),
		"reference_only_games": ref_only, "other_only_games": other_only,
		"diff_win_rate": (len(other_only) - len(ref_only)) / n if n else 0.0,
		"p_value": sign_test_p(len(ref_only), len(other_only)),
	}


def pct(x: float) -> str:
	return f"{100 * x:5.1f}%"


def print_report(result: dict) -> None:
	base = result["chance_baseline"]
	ref = result["reference"]
	print(f"\nGames: {result['games']}  seed: {result['seed']}  chance baseline: {pct(base)}")
	names = sorted(result["contenders"], key=lambda c: -result["contenders"][c]["win_rate"])
	print(f"\n{'contender':<34}{'wins':>6}{'draws':>7}{'win rate':>10}   95% CI")
	for c in names:
		s = result["contenders"][c]
		print(f"{c:<34}{s['wins']:>6}{s['draws']:>7}{pct(s['win_rate']):>10}   "
			f"[{pct(s['ci95'][0])}, {pct(s['ci95'][1])}]")
	counts = sorted({k for s in result["contenders"].values() for k in s["by_players"]}, key=int)
	print("\nWin rate by player count (wins/games)")
	print(f"{'contender':<34}" + "".join(f"{k + 'p':>14}" for k in counts))
	for c in names:
		row = ""
		for k in counts:
			e = result["contenders"][c]["by_players"].get(k)
			row += f"{(f'{pct(e['win_rate'])} {e['wins']}/{e['games']}' if e else '-'):>14}"
		print(f"{c:<34}{row}")
	bots = sorted({k for s in result["contenders"].values() for k in s["by_opponent_bot"]})
	print("\nWin rate by opponent bot present at the table")
	short = [f"opp{i + 1}" for i in range(len(bots))]
	for sh, b in zip(short, bots):
		print(f"  {sh} = {b}")
	print(f"{'contender':<34}" + "".join(f"{s:>13}" for s in short))
	for c in names:
		row = ""
		for b in bots:
			e = result["contenders"][c]["by_opponent_bot"].get(b)
			row += f"{(pct(e['win_rate']) if e else '-'):>13}"
		print(f"{c:<34}{row}")
	if len(names) > 1:
		print(f"\nPaired vs reference '{ref}' (+ = contender better)")
		print(f"{'contender':<34}{'ref-only':>9}{'other-only':>11}{'diff':>9}{'p':>9}")
		for c in names:
			if c == ref:
				continue
			p = result["paired"][c]
			print(f"{c:<34}{p['reference_only_wins']:>9}{p['other_only_wins']:>11}"
				f"{100 * p['diff_win_rate']:>+8.1f}%{p['p_value']:>9.4f}")


def save_plot(result: dict, path: str) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	cs = result["contenders"]
	names = sorted(cs, key=lambda c: cs[c]["win_rate"])
	rates = [100 * cs[c]["win_rate"] for c in names]
	lo = [r - 100 * cs[c]["ci95"][0] for r, c in zip(rates, names)]
	hi = [100 * cs[c]["ci95"][1] - r for r, c in zip(rates, names)]
	fig, ax = plt.subplots(figsize=(8, 0.4 * len(names) + 1.8))
	ax.barh(names, rates, xerr=[lo, hi], color="#4c78a8", capsize=3)
	ax.axvline(100 * result["chance_baseline"], color="#555", linestyle="--",
		label=f"chance ({100 * result['chance_baseline']:.1f}%)")
	ax.set_xlabel("win rate (%) with 95% Wilson interval")
	ax.legend(loc="lower right")
	fig.tight_layout()
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	fig.savefig(path, dpi=130)
	plt.close(fig)


# ----------------------------------------------------------------------------
def build_contenders(args, parser) -> list[tuple[str, callable]]:
	out: list[tuple[str, callable]] = []
	for item in args.contenders:
		if "=" not in item:
			parser.error(f"contender must be name=path, got {item!r}")
		name, path = item.split("=", 1)

		def factory(path=path):
			return LLMAgent(
				CONTENDER_ID, name=CONTENDER_ID, seed=0, model_path=path,
				temperature=args.temperature, greedy=args.greedy,
				history_max_lines=getattr(args, "history_max_lines", 600),
			)
		out.append((name, factory))
	if args.bots:
		by_name = {c.__name__: c for c in TEACHERS}
		wanted = list(by_name) if args.bots.lower() == "all" else [
			b.strip() for b in args.bots.split(",") if b.strip()]
		for b in wanted:
			if b not in by_name:
				parser.error(f"unknown bot {b!r}; choose from {', '.join(by_name)}")
			out.append((b, lambda cls=by_name[b]: make_player(cls, CONTENDER_ID, CONTENDER_ID, 0)))
	names = [n for n, _ in out]
	if len(set(names)) != len(names):
		parser.error("contender names must be unique")
	if not out:
		parser.error("no contenders given (use --contenders and/or --bots)")
	return out


def compare(contenders, seed, games, counts, pool_size, reference=None, log=print,
		progress_every: int = 25) -> dict:
	"""contenders: list of (name, factory). Returns the JSON-ready result dict."""
	spec, slots = make_schedule(seed, games, counts, pool_size)
	results = {}
	for name, factory in contenders:
		records = run_contender(name, factory, spec, slots, progress_every, log)
		results[name] = summarize(records, spec, slots)
	reference = reference or contenders[0][0]
	if reference not in results:
		raise ValueError(f"reference {reference!r} is not a contender")
	return {
		"seed": seed, "games": games, "players": counts, "pool_size": pool_size,
		"chance_baseline": sum(1 / s.n_players for s in slots) / max(len(slots), 1),
		"max_turns_per_player": MAX_TURNS_PER_PLAYER,
		"pool": [pid for _, pid, _ in spec],
		"schedule": [{
			"game": s.index, "n_players": s.n_players, "contender_seat": s.contender_seat,
			"seats": [spec[x][1] if x >= 0 else CONTENDER_ID for x in s.seats],
		} for s in slots],
		"reference": reference,
		"contenders": results,
		"paired": {c: paired(results[reference], results[c]) for c in results if c != reference},
	}


def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--contenders", nargs="*", default=[], metavar="NAME=PATH",
		help="LM contenders, each name=checkpoint_path")
	p.add_argument("--bots", default=None,
		help="'all' or a comma list of teacher bot class names")
	p.add_argument("--games", type=int, default=300)
	p.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6])
	p.add_argument("--pool-size", type=int, default=6)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--greedy", action="store_true")
	p.add_argument("--temperature", type=float, default=1.0)
	p.add_argument("--history-max-lines", type=int, default=600,
		help="LM contenders' past-game history budget (lm.rl trains with 200)")
	p.add_argument("--reference", default=None, help="default: first contender")
	p.add_argument("--out", default="data/compare.json")
	p.add_argument("--plot", default=None, help="optional PNG path")
	args = p.parse_args(argv)
	if any(not 2 <= k <= 6 for k in args.players):
		p.error("player counts must be in 2..6")

	contenders = build_contenders(args, p)
	if args.reference and args.reference not in [n for n, _ in contenders]:
		p.error(f"--reference {args.reference!r} is not a contender")
	result = compare(contenders, args.seed, args.games, args.players, args.pool_size,
		args.reference)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=1)
	print_report(result)
	if args.plot:
		save_plot(result, args.plot)
		print(f"\nplot -> {args.plot}")
	print(f"results -> {args.out}")


if __name__ == "__main__":
	main()
