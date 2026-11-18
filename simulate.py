from agents.aggressive_belief import AggressiveBelief
from agents.always_duke import AlwaysDuke
from agents.belief_honest_random import BeliefHonestRandom
from agents.duke_fish import DukeFish
from agents.final_agent import FinalAgent
from agents.hibernate import Hibernate
from agents.honest_random import HonestRandom
from agents.probabilistic_belief_tweaker import ProbabilisticBeliefTweaker
from agents.random_agent import RandomAgent
from agents.passive_agent import PassiveAgent
from agents.probabilistic_belief_honest_random import ProbabilisticBeliefHonestRandom
from agents.aggressive_probabilistic_belief import AggressiveProbabilisticBelief
from agents.probabilistic_belief_duke_fish import ProbabilsticBeliefDukeFish
from src.game import Game

import random

MAX_TURNS_PER_PLAYER = 100

def _play_game(game: Game) -> bool:
    turns_taken = {player.id: 0 for player in game.players}

    while game.winner is None:
        player = game.current
        player.choose_action(game)
        turns_taken[player.id] += 1

        if game.winner is None and all(
            turns_taken[player.id] >= MAX_TURNS_PER_PLAYER
            for player in game.alive_players
        ):
            return True

    return False


def main(games: int = 10000):
    final_agent = FinalAgent("Final Agent")
    playerls = [
        AlwaysDuke("Always Duke"),
        PassiveAgent("Passive Agent"),
        AggressiveBelief("Aggressive Belief"),
        # AggressiveBelief("Aggressive Belief 2"),
        BeliefHonestRandom("Belief Honest Random"),
        DukeFish("Duke Fish"),
        HonestRandom("Honest Random"),
        RandomAgent("Random Agent"),
        ProbabilisticBeliefHonestRandom("Probabilistic Belief Honest Random"),
        # ProbabilisticBeliefHonestRandom("Probabilistic Belief Honest Random 2"),
        AggressiveProbabilisticBelief("Aggressive Probabilistic Belief"),
        # AggressiveProbabilisticBelief("Aggressive Probabilistic Belief 2"),
        ProbabilisticBeliefTweaker("Probabilistic Belief Tweaker"),
        ProbabilsticBeliefDukeFish("Probabilistic Belief Duke Fish"),
        # Hibernate("Hibernate"),
        final_agent,
    ]

    winners = {
        player.name: [0, 0] for player in playerls
        }
    points = {player.name: 0.0 for player in playerls}
    draws = 0
    tweaker_endgame_count = 0

    for _ in range(games):
        players = random.sample(playerls, 4)
        for i in players:
            winners[i.name][1] += 1
        random.shuffle(players)
        game = Game(players)
        # displayed_log_length = 0

        if _play_game(game):
            survivors = game.alive_players
            points_per_player = 1 / len(survivors)
            for player in survivors:
                points[player.name] += points_per_player
            for player in game.players:
                player.on_game_end(None)
            draws += 1
            print(
                "\nDraw: "
                f"{', '.join(player.name for player in survivors)} "
                f"split 1 point"
            )
        else:
            winner = game.winner
            assert winner is not None
            print(f"\nWinner: {winner.name}")
            winners[winner.name][0] += 1
            points[winner.name] += 1
            # for event in game.log[displayed_log_length:]:
            #     print(f"  {event}")
            # displayed_log_length = len(game.log)

    print("\nNumber of wins:", winners)
    print(f"Draws: {draws}")
    print("Points:", points)
    print("Win percentages")
    for i in winners:
        print(f"{i}: {winners[i][0]/winners[i][1]*100:.2f}% ({winners[i][0]}/{winners[i][1]})")
    final_agent_games = winners[final_agent.name][1]
    endgame_percentage = (
        final_agent._endgame_count / final_agent_games * 100
        if final_agent_games
        else 0
    )
    print(
        f"Final Agent reached the endgame: "
        f"{final_agent._endgame_count}/{final_agent_games} games "
        f"({endgame_percentage:.2f}%)"
    )


if __name__ == "__main__":
    main()