# Agent submissions

Each submission is one standalone Python file. The tournament starts a fresh
Python process for each action or response and exchanges one JSON request and
one JSON response over standard input/output. A bot must finish each response
within 1.95 seconds. It may import Python standard-library modules, but not
third-party or project modules.

Copy [`agents/starter_agent.py`](agents/starter_agent.py) and edit it to create
an agent. Place `.py` submissions in a directory and run a tournament with:

```bash
python run_tournament.py --submission-dir path/to/agents
```

Each `.py` file in that directory is loaded as one tournament entry. The
starter is included when `submission/agents/` is used as the directory.

## Request

Every request is a JSON object with:

| Field | Description |
| --- | --- |
| `protocol_version` | Protocol version, currently `1`. |
| `request_id` | Unique identifier for this request. |
| `agent_id` | The agent's player ID in this game. |
| `game_id` | Unique ID for this game. |
| `phase` | `action`, `challenge`, `block`, `reveal_influence`, or `exchange`. |
| `state` | Current serializable game state from this agent's point of view. |
| `memory` | The last JSON value returned by this agent, or `null`. |
| `legal_actions` | Present for `action`; exact permitted `{name, target_id}` objects. |
| `context` | Present for decisions; the decision-specific public information. |
| `legal_decisions` | Present for decisions; the exact permitted decision values. |

`state.players` contains every player's public coins and influence. The agent's
own cards and all revealed cards are visible; unrevealed opponents' cards are
shown as `"hidden"`. `state.log` contains the public game history. The supplied
legal options are authoritative: an action or decision not listed is an
automatic forfeit.

## Response

Return exactly one JSON object on standard output. Do not print logs or
diagnostics to standard output; use standard error for debugging.

For an action:

```json
{"action": {"name": "income", "target_id": null}, "memory": {"games": 1}}
```

The `action` object must exactly match an entry in `legal_actions`.

For a decision:

```json
{"decision": false, "memory": {"games": 1}}
```

`decision` must exactly match an entry in `legal_decisions`. Return a JSON
value in `memory` on every response. It is limited to 64 KiB and is sent back
on the next action or decision, including across games in the same tournament
run. Use `game_id` to distinguish a new game and reset any game-specific
memory. Only the JSON memory value persists; process globals do not.

## Automatic elimination

A process crash, timeout, malformed response, oversized memory value, illegal
action, or illegal decision immediately forfeits the agent. It is eliminated
from the game and the reason is added to the game history. Actions are checked
against the engine's legal options before execution.

## Validation and hosting

The runner checks Python syntax and rejects direct imports outside the Python
standard library. This check is not a security boundary: Python code can
dynamically import modules or access host resources. `subprocess` plus a
timeout also does not sandbox untrusted code. Do not accept public uploads on a
host that runs these files directly. A public tournament service must execute
each submission in a separately configured OS/container sandbox with network,
filesystem, process, memory, and CPU restrictions. The repository currently
provides the protocol and local tournament adapter, not that hosting sandbox
or an upload website.
