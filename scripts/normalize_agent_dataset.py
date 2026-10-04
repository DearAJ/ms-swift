import json
import sys

src, dst = sys.argv[1], sys.argv[2]

QUERY_ROLES = {"user", "tool"}
RESPONSE_ROLES = {"assistant"}
DEFAULT_LOSS_TRUE_ROLES = {"assistant", "tool_call"}


def normalize_messages(messages):
    normalized = []
    for message in messages:
        message = dict(message)
        role = message.get("role")
        if "loss" not in message:
            message["loss"] = role in DEFAULT_LOSS_TRUE_ROLES
        normalized.append(message)
    return normalized


def template_compatible(messages):
    roles = [message.get("role") for message in messages]
    if roles and roles[0] == "system":
        roles = roles[1:]

    # The swift template folds tool_call messages into assistant messages before
    # checking the final user/tool -> assistant turn structure.
    compact = []
    i = 0
    while i < len(roles):
        role = roles[i]
        if role == "tool_call":
            compact.append("assistant")
            while i + 1 < len(roles) and roles[i + 1] == "tool_call":
                i += 1
        else:
            compact.append(role)
        i += 1

    roles = []
    i = 0
    while i < len(compact):
        role = compact[i]
        prev = roles[-1] if roles else None
        if prev == "assistant" and role == "tool":
            roles.append("tool")
            while i + 1 < len(compact) and compact[i + 1] == "tool":
                i += 1
        elif (prev == "assistant" and role == "assistant") or (prev == "user" and role == "user"):
            pass
        elif role == "tool" and prev not in {"assistant", "tool"}:
            pass
        else:
            roles.append(role)
        i += 1

    if roles and len(roles) % 2 == 1:
        roles.append("assistant")
    return all(
        role in (QUERY_ROLES if i % 2 == 0 else RESPONSE_ROLES)
        for i, role in enumerate(roles)
    )


written = 0
skipped = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, 1):
        item = json.loads(line)
        item["messages"] = normalize_messages(item.get("messages", []))
        if not template_compatible(item["messages"]):
            skipped += 1
            continue
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1

print(f"normalized dataset written to {dst}: {written} samples, skipped {skipped}")
