import argparse
import random
from pathlib import Path

from agents.aggressive_belief import AggressiveBelief
from agents.aggressive_probabilistic_belief import AggressiveProbabilisticBelief
from agents.always_duke import AlwaysDuke
from agents.belief_honest_random import BeliefHonestRandom
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.hibernate import Hibernate
from agents.honest_random import HonestRandom
from agents.param_bot import ParamBot
from agents.passive_agent import PassiveAgent
from agents.probabilistic_belief_duke_fish import ProbabilsticBeliefDukeFish
from agents.probabilistic_belief_honest_random import (
    ProbabilisticBeliefHonestRandom,
)
from agents.probabilistic_belief_tweaker import ProbabilisticBeliefTweaker
from agents.random_agent import RandomAgent
from src.player import Player
from submission import create_submission_agents
from tournament import TournamentManager


def create_agents() -> list[Player]:
    return [
        AggressiveBelief("aggressive", "Aggressive Belief"),
        AggressiveProbabilisticBelief(
            "aggressive-prob", "Aggressive Probabilistic Belief"
        ),
        AlwaysDuke("duke", "Always Duke"),
        BeliefHonestRandom("belief-honest", "Belief Honest Random"),
        DukeFish("duke-fish", "Duke Fish"),
        FinalAgent("final", "Final Agent"),
        # Hibernate("hibernate", "Hibernate"),
        HonestRandom("honest", "Honest Random"),
        ParamBot("param", "ParamBot"),
        # PassiveAgent("passive", "Passive Agent"),
        ProbabilsticBeliefDukeFish(
            "prob-duke-fish", "Probabilistic Duke Fish"
        ),
        ProbabilisticBeliefHonestRandom(
            "prob-honest", "Probabilistic Belief Honest Random"
        ),
        ProbabilisticBeliefTweaker("tweaker", "Probabilistic Belief Tweaker"),
        RandomAgent("random", "Random Agent"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Coup tournament with the built-in agents."
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--games-per-group", type=int, default=1_000)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--submission-dir",
        type=Path,
        help="Add each .py agent in this directory to the built-in roster.",
    )
    args = parser.parse_args()

    players = create_agents()
    if args.submission_dir is not None:
        players.extend(create_submission_agents(args.submission_dir))

    tournament = TournamentManager(
        players,
        rounds=args.rounds,
        games_per_group=args.games_per_group,
        max_turns=args.max_turns,
        rng=random.Random(args.seed),
    )
    result = tournament.run()

    print(
        f"Tournament complete: {result.rounds} rounds, "
        f"{tournament.games_per_group:,} games per group"
    )
    print("\nFinal leaderboard:")
    for entry in result.leaderboard:
        print(
            f"{entry.rank:>2}. {entry.name}: {entry.points} points, "
            f"{entry.wins} wins, {entry.draws} draws, {entry.byes} byes"
        )
    if result.playoff_wins:
        print("\nPlayoff wins:")
        for player_id, wins in sorted(
            result.playoff_wins.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            name = result.standings[player_id].player.name
            print(f"  {name}: {wins}")


if __name__ == "__main__":
    main()
