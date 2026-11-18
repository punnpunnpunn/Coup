import json
import sys


def respond(request):
    phase = request["phase"]
    memory = request.get("memory") or {"turns": 0}
    memory["turns"] = memory.get("turns", 0) + (phase == "action")

    if phase == "action":
        choices = request["legal_actions"]
        coup = [choice for choice in choices if choice["name"] == "coup"]
        action = coup[0] if coup else next(
            choice for choice in choices if choice["name"] == "income"
        )
        return {"action": action, "memory": memory}

    if phase == "challenge":
        decision = False
    elif phase == "block":
        decision = None
    elif phase == "reveal_influence":
        decision = request["legal_decisions"][0]
    elif phase == "exchange":
        decision = request["legal_decisions"][0]
    else:
        raise ValueError("Unknown phase")
    return {"decision": decision, "memory": memory}


if __name__ == "__main__":
    print(json.dumps(respond(json.load(sys.stdin))))
