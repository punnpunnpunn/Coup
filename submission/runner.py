from __future__ import annotations

import ast
import json
import subprocess
import sys
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any

from src.game import Game
from src.player import Player


MAX_MOVE_SECONDS = 1.95
MAX_MEMORY_BYTES = 64 * 1024
PROTOCOL_VERSION = 1
DEFAULT_SUBMISSION_DIR = Path(__file__).parent / "agents"
_ACTIONS = (
    "income",
    "foreign_aid",
    "coup",
    "tax",
    "assassinate",
    "exchange",
    "steal",
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


class SubmissionError(ValueError):
    """A submitted program failed the agent protocol."""


def validate_source(source: str | Path) -> Path:
    path = Path(source).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("A submission must be a Python source file.")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"Could not read valid Python source: {error}") from error

    unavailable = sorted(
        {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.split(".", 1)[0] not in sys.stdlib_module_names
        }
        | {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.level == 0
            and node.module.split(".", 1)[0] not in sys.stdlib_module_names
        }
    )
    relative_imports = any(
        isinstance(node, ast.ImportFrom) and node.level > 0
        for node in ast.walk(tree)
    )
    if unavailable or relative_imports:
        rejected = unavailable + (["relative imports"] if relative_imports else [])
        raise ValueError(
            "Submissions may only import Python standard-library modules; "
            f"unsupported import(s): {', '.join(rejected)}."
        )
    return path


def create_submission_agents(
    directory: str | Path = DEFAULT_SUBMISSION_DIR,
    *,
    timeout_seconds: float = MAX_MOVE_SECONDS,
) -> list[SubmissionAgent]:
    folder = Path(directory)
    if not folder.is_dir():
        raise ValueError(f"Submission directory does not exist: {folder}")
    paths = sorted(folder.glob("*.py"))
    if not paths:
        raise ValueError(f"No Python submissions found in {folder}.")
    return [
        SubmissionAgent(
            player_id=f"submission-{path.stem}",
            name=path.stem.replace("_", " "),
            source=path,
            timeout_seconds=timeout_seconds,
        )
        for path in paths
    ]


class SubmissionAgent(Player):
    def __init__(
        self,
        player_id: str,
        name: str,
        source: str | Path,
        *,
        timeout_seconds: float = MAX_MOVE_SECONDS,
    ) -> None:
        super().__init__(player_id, name)
        if not 0 < timeout_seconds <= MAX_MOVE_SECONDS:
            raise ValueError(
                f"Move timeout must be greater than zero and at most "
                f"{MAX_MOVE_SECONDS} seconds."
            )
        self.source = validate_source(source)
        self.timeout_seconds = timeout_seconds
        self.memory: Any = None
        self._game_id: str | None = None
        self._disqualified = False

    def choose_action(self, game: Game) -> dict[str, Any]:
        if game.current.id != self.id:
            raise ValueError("It is not this agent's turn.")
        if self._disqualified:
            return game.state_for(self.id)

        self._begin_game(game)
        legal_actions = self._legal_actions(game)
        try:
            response = self._request(
                phase="action",
                state=game.state_for(self.id),
                legal_actions=legal_actions,
            )
            action = response.get("action")
            if not isinstance(action, dict) or action not in legal_actions:
                raise SubmissionError("invalid_response")
            self.memory = response["memory"]
        except SubmissionError as error:
            self._forfeit(game, str(error), advance_turn=True)
            return game.state_for(self.id)

        return game.perform_action(
            self.id, action["name"], target_id=action["target_id"]
        )

    def decide(self, kind: str, context: dict[str, object]) -> object:
        game = getattr(self, "_active_game", None)
        if game is None:
            raise RuntimeError("Submission decision requested outside a game.")
        return self._decide_for_game(game, kind, context)

    def decide_with_game(
        self, game: Game, kind: str, context: dict[str, object]
    ) -> object:
        self._active_game = game
        try:
            return self._decide_for_game(game, kind, context)
        finally:
            self._active_game = None

    def _decide_for_game(
        self, game: Game, kind: str, context: dict[str, object]
    ) -> object:
        if self._disqualified:
            return self._fallback_decision(kind, context)

        self._begin_game(game)
        legal_decisions = self._legal_decisions(kind, context)
        if legal_decisions is None:
            self._forfeit(game, "invalid_decision_context")
            return self._fallback_decision(kind, context)

        try:
            response = self._request(
                phase=kind,
                state=game.state_for(self.id),
                context=context,
                legal_decisions=legal_decisions,
            )
            decision = response.get("decision")
            if not self._is_legal_decision(kind, decision, legal_decisions):
                raise SubmissionError("invalid_response")
            self.memory = response["memory"]
        except SubmissionError as error:
            self._forfeit(game, str(error))
            return self._fallback_decision(kind, context, legal_decisions)

        if kind == "block" and decision is not None:
            if context["action"] == "foreign_aid":
                return {"blocker_id": self.id}
            return {
                "blocker_id": self.id,
                "role": decision["role"],
            }
        return decision

    def _begin_game(self, game: Game) -> None:
        if self._game_id != game.game_id:
            self._game_id = game.game_id
            self._disqualified = False

    def _request(
        self,
        *,
        phase: str,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None = None,
        context: dict[str, object] | None = None,
        legal_decisions: list[Any] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": uuid.uuid4().hex,
            "agent_id": self.id,
            "game_id": self._game_id,
            "phase": phase,
            "state": state,
            "memory": self.memory,
        }
        if legal_actions is not None:
            request["legal_actions"] = legal_actions
        if context is not None:
            request["context"] = context
        if legal_decisions is not None:
            request["legal_decisions"] = legal_decisions

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(self.source)],
                input=json.dumps(request, separators=(",", ":")),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=self.source.parent,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise SubmissionError("timeout") from error
        except OSError as error:
            raise SubmissionError("crash") from error
        if completed.returncode != 0:
            raise SubmissionError("crash")
        try:
            response = json.loads(
                completed.stdout,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as error:
            raise SubmissionError("invalid_response") from error
        if (
            not isinstance(response, dict)
            or "memory" not in response
            or not self._valid_memory(response["memory"])
        ):
            raise SubmissionError("invalid_response")
        return response

    @staticmethod
    def _valid_memory(memory: Any) -> bool:
        try:
            encoded = json.dumps(memory, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            return False
        return len(encoded) <= MAX_MEMORY_BYTES

    def _legal_actions(self, game: Game) -> list[dict[str, Any]]:
        opponents = [
            player.id for player in game.alive_players if player.id != self.id
        ]
        if self.coins >= 10:
            names = ("coup",)
        else:
            names = tuple(
                name
                for name in _ACTIONS
                if (name != "coup" or self.coins >= 7)
                and (name != "assassinate" or self.coins >= 3)
            )
        actions = []
        for name in names:
            targets = opponents if name in ("coup", "assassinate", "steal") else [None]
            actions.extend(
                {"name": name, "target_id": target_id} for target_id in targets
            )
        return actions

    def _legal_decisions(
        self, kind: str, context: dict[str, object]
    ) -> list[Any] | None:
        if kind == "challenge":
            return [False, True]
        if kind == "block":
            legal = [None]
            if self.id not in context["eligible_blockers"]:
                return legal
            action = context["action"]
            if action == "foreign_aid":
                legal.append({"role": "duke"})
            else:
                legal.extend(
                    {"role": role} for role in context["allowed_roles"]
                )
            return legal
        if kind == "reveal_influence":
            return [card["index"] for card in context["cards"]]
        if kind == "exchange":
            cards = context["cards"]
            count = context["keep_count"]
            return [
                list(indices)
                for indices in combinations(range(len(cards)), count)
            ] if 0 <= count <= len(cards) else None
        return None

    @staticmethod
    def _is_legal_decision(
        kind: str, decision: Any, legal_decisions: list[Any]
    ) -> bool:
        if kind == "challenge":
            return isinstance(decision, bool) and decision in legal_decisions
        if kind == "reveal_influence":
            return (
                isinstance(decision, int)
                and not isinstance(decision, bool)
                and decision in legal_decisions
            )
        if kind == "exchange":
            return (
                isinstance(decision, list)
                and all(
                    isinstance(index, int) and not isinstance(index, bool)
                    for index in decision
                )
                and decision in legal_decisions
                and len(set(decision)) == len(decision)
            )
        if kind == "block":
            if decision is None:
                return True
            if not isinstance(decision, dict) or len(decision) != 1:
                return False
            role = decision.get("role")
            return {"role": role} in legal_decisions
        return False

    @staticmethod
    def _fallback_decision(
        kind: str,
        context: dict[str, object],
        legal_decisions: list[Any] | None = None,
    ) -> object:
        if legal_decisions:
            return legal_decisions[0]
        if kind == "challenge":
            return False
        if kind == "block":
            return None
        if kind == "reveal_influence":
            cards = context.get("cards", [])
            return cards[0]["index"] if cards else 0
        if kind == "exchange":
            return list(range(context.get("keep_count", 0)))
        raise ValueError(f"Unknown decision type: {kind}")

    def _forfeit(
        self, game: Game, reason: str, *, advance_turn: bool = False
    ) -> None:
        if self._disqualified:
            return
        self._disqualified = True
        game.forfeit_player(self.id, reason)
        if advance_turn and game.current.id == self.id:
            game.end_turn()
