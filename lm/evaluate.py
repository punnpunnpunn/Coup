"""Evaluate an LLMAgent against the scripted bots.

Usage: python -m lm.evaluate --games 200 --model checkpoints/coup-lm [--greedy]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import random
import time

from agents.aggressive_belief import AggressiveBelief
from agents.aggressive_probabilistic_belief import AggressiveProbabilisticBelief
from agents.always_duke import AlwaysDuke
from agents.belief_honest_random import BeliefHonestRandom
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.hibernate import Hibernate
from agents.honest_random import HonestRandom
from agents.llm_agent import LLMAgent
from agents.passive_agent import PassiveAgent
from agents.probabilistic_belief_honest_random import ProbabilisticBeliefHonestRandom
from agents.random_agent import RandomAgent
from src.game import Game

TEACHERS = [
	AlwaysDuke, PassiveAgent, AggressiveBelief, BeliefHonestRandom, DukeFish,
	HonestRandom, RandomAgent, ProbabilisticBeliefHonestRandom,
	AggressiveProbabilisticBelief, Hibernate, FinalAgent,
]
MAX_TURNS_PER_PLAYER = 100


def play(game: Game) -> bool:
	"""Play to the end; return True if the turn cap produced a draw."""
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


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--games", type=int, default=200)
	p.add_argument("--model", default=None)
	p.add_argument("--greedy", action="store_true")
	p.add_argument("--temperature", type=float, default=1.0)
	p.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6],
		help="player counts to sample from")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--pool-size", type=int, default=6, help="number of persistent opponent bots")
	p.add_argument("--fresh", action="store_true",
		help="ablation: new LLMAgent (empty history) every game")
	args = p.parse_args()

	rng = random.Random(args.seed)
	wins: dict[int, list[int]] = {}
	draws = 0
	t0 = time.time()
	# Persistent opponents and (unless --fresh) a persistent LLM agent, so the
	# agent's cross-game history is about opponents it keeps meeting.
	pool = [_make_bot(rng, i) for i in range(args.pool_size)]

	def new_llm():
		return LLMAgent(
			"LLM", name="LLM", seed=rng.randrange(2**31), model_path=args.model,
			temperature=args.temperature, greedy=args.greedy,
		)

	llm = new_llm()
	for g in range(args.games):
		k = min(rng.choice(args.players), len(pool) + 1)
		if args.fresh:
			llm = new_llm()
		players = [llm] + rng.sample(pool, k - 1)
		rng.shuffle(players)
		with contextlib.redirect_stdout(io.StringIO()):
			game = Game(players)
			draw = play(game)
		stat = wins.setdefault(k, [0, 0])
		stat[1] += 1
		if draw:
			draws += 1
		elif game.winner is llm:
			stat[0] += 1
		if (g + 1) % 10 == 0:
			print(f"{g + 1}/{args.games} games, {time.time() - t0:.0f}s", flush=True)

	total_w = sum(w for w, _ in wins.values())
	total_n = sum(n for _, n in wins.values())
	baseline = sum(n / k for k, (_, n) in wins.items())
	print(f"\nLLM wins: {total_w}/{total_n} = {100 * total_w / max(total_n, 1):.1f}% "
		f"(uniform baseline {100 * baseline / max(total_n, 1):.1f}%), draws: {draws}")
	for k in sorted(wins):
		w, n = wins[k]
		print(f"  {k} players: {w}/{n} = {100 * w / max(n, 1):.1f}% (baseline {100 / k:.1f}%)")


def _make_bot(rng, i):
	cls = rng.choice(TEACHERS)
	name = f"{cls.__name__} {i}"
	if "seed" in cls.__init__.__code__.co_varnames:
		return cls(name, seed=rng.randrange(2**31))
	return cls(name)


if __name__ == "__main__":
	main()
