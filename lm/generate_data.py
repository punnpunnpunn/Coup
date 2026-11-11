"""Generate text transcripts of bot-vs-bot Coup games.

Usage: python -m lm.generate_data --games 20000 --out data/transcripts.jsonl
"""
import argparse
import contextlib
import inspect
import io
import json
import os
import random
import sys

from agents.aggressive_belief import AggressiveBelief
from agents.aggressive_probabilistic_belief import AggressiveProbabilisticBelief
from agents.always_duke import AlwaysDuke
from agents.belief_honest_random import BeliefHonestRandom
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.hibernate import Hibernate
from agents.honest_random import HonestRandom
from agents.passive_agent import PassiveAgent
from agents.probabilistic_belief_honest_random import ProbabilisticBeliefHonestRandom
from agents.random_agent import RandomAgent
from lm.encoding import History, record
from src.game import Game

TEACHERS = [
	AlwaysDuke,
	PassiveAgent,
	AggressiveBelief,
	BeliefHonestRandom,
	DukeFish,
	HonestRandom,
	RandomAgent,
	ProbabilisticBeliefHonestRandom,
	AggressiveProbabilisticBelief,
	Hibernate,
	FinalAgent,
]

MAX_TURNS_PER_PLAYER = 100


def make_player(cls, player_id: str, name: str, seed: int):
	if "seed" in inspect.signature(cls.__init__).parameters:
		return cls(player_id, name=name, seed=seed)
	return cls(player_id, name=name)


def play_game(game: Game) -> bool:
	"""Play to completion. Returns True if the turn cap was hit (draw)."""
	turns = {p.id: 0 for p in game.players}
	while game.winner is None:
		player = game.current
		player.choose_action(game)
		turns[player.id] += 1
		if game.winner is None and all(
			turns[p.id] >= MAX_TURNS_PER_PLAYER for p in game.alive_players
		):
			return True
	return False



def play_session_game(rng, pool, histories, n_players, session, game_index):
	"""Play one recorded game among a sample of the session pool.

	pool: list of (cls, player_id, name). histories: player_id -> History (persist
	across the session). Returns (records or None on a draw).
	"""
	chosen = rng.sample(pool, n_players)
	rng.shuffle(chosen)
	players = [make_player(cls, pid, name, rng.randrange(2**31)) for cls, pid, name in chosen]
	random.seed(rng.randrange(2**31))  # the engine shuffles with the global RNG
	game = Game(players)
	transcripts = [record(p, game, histories[p.id]) for p in players]
	with contextlib.redirect_stdout(io.StringIO()):
		draw = play_game(game)
	winner_id = None if draw else game.winner.id
	ids = [p.id for p in players]
	records = []
	for p, t in zip(players, transcripts):
		if not draw:
			records.append({
				"n": n_players,
				"result": "win" if p.id == winner_id else "lose",
				"teacher": type(p).__name__,
				"session": session,
				"game_index": game_index,
				"history": t.history_lines(),
				"body": t.body,
			})
		histories[p.id].add_game(game.events, ids, winner_id)
	return None if draw else records


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--games", type=int, default=20000)
	parser.add_argument("--out", default="data/transcripts.jsonl")
	parser.add_argument("--min-players", type=int, default=2)
	parser.add_argument("--max-players", type=int, default=6)
	parser.add_argument("--min-pool", type=int, default=3)
	parser.add_argument("--max-pool", type=int, default=7)
	parser.add_argument("--games-per-session", type=int, default=10)
	parser.add_argument("--history-lines", type=int, default=600)
	parser.add_argument("--history-games", type=int, default=8)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args(argv)
	if not 2 <= args.min_players <= args.max_players <= 6:
		parser.error("player counts must satisfy 2 <= min <= max <= 6")
	if not args.min_players <= args.min_pool <= args.max_pool:
		parser.error("pool sizes must satisfy min-players <= min-pool <= max-pool")

	rng = random.Random(args.seed)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

	draws = errors = written = lines = hist_lines = games_done = session = 0
	with open(args.out, "w", encoding="utf-8") as f:
		while games_done < args.games:
			pool_size = rng.randint(args.min_pool, args.max_pool)
			pool = []
			for i in range(pool_size):
				cls = rng.choice(TEACHERS)
				pool.append((cls, f"{cls.__name__} {i}", f"{cls.__name__} {i}"))
			histories = {
				pid: History(pid, args.history_lines, args.history_games) for _, pid, _ in pool
			}
			for game_index in range(min(args.games_per_session, args.games - games_done)):
				games_done += 1
				n = rng.randint(args.min_players, min(args.max_players, pool_size))
				try:
					records = play_session_game(rng, pool, histories, n, session, game_index)
				except Exception as exc:  # keep long runs alive; report at the end
					errors += 1
					if errors <= 5:
						print(f"game {games_done} failed: {exc!r}", file=sys.stderr)
					continue
				if records is None:
					draws += 1
				else:
					for rec in records:
						f.write(json.dumps(rec) + "\n")
						lines += len(rec["body"])
						hist_lines += len(rec["history"])
						written += 1
				if games_done % 1000 == 0:
					print(f"{games_done}/{args.games} games, {written} transcripts", flush=True)
			session += 1

	avg = lines / written if written else 0.0
	avg_h = hist_lines / written if written else 0.0
	print(
		f"done: {games_done} games in {session} sessions ({draws} draws discarded, "
		f"{errors} errors), {written} transcripts, avg {avg:.1f} body lines, "
		f"avg {avg_h:.1f} history lines -> {args.out}"
	)


if __name__ == "__main__":
	main()
