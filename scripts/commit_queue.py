"""Apply one reviewed, checksum-verified change after two hours have elapsed."""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

STATE = ".github/commit-plan.json"
ALLOWED = {"SqueezeNet_EuroSAT.py", "README.md"}
INTERVAL = 2 * 60 * 60


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_queue(root, queue_path):
    state = json.loads((root / STATE).read_text(encoding="utf-8"))
    raw = queue_path.read_bytes()
    if digest(raw) != state["queue_sha256"]:
        raise ValueError("Queue checksum does not match the reviewed plan")
    queue = json.loads(raw)
    if len(queue) != 9 or [entry["number"] for entry in queue] != list(range(2, 11)):
        raise ValueError("Expected exactly nine ordered changes (commits 2 through 10)")
    for entry in queue:
        if not entry["message"] or "\n" in entry["message"]:
            raise ValueError("Invalid commit message")
        if not entry["files"] or set(entry["files"]) - ALLOWED:
            raise ValueError("Queue contains an unexpected path")
        if set(entry["before"]) != set(entry["files"]):
            raise ValueError("Missing expected prior file checksums")
        for name, content in entry["files"].items():
            if name.endswith(".py"):
                ast.parse(content, filename=name)
    return state, queue


def emit(name, value):
    print(f"{name}={value}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def status(root, now=None):
    state = json.loads((root / STATE).read_text(encoding="utf-8"))
    completed = state["completed"]
    if not isinstance(completed, int) or not 1 <= completed <= 10:
        raise ValueError("Invalid completed-commit count")
    last = int(git(root, "log", "-1", "--format=%ct", "--", STATE))
    now = time.time() if now is None else now
    return completed == 10, completed < 10 and now - last >= INTERVAL


def apply_next(root, queue_path, now=None):
    done, due = status(root, now)
    if done or not due:
        print("All ten commits are complete." if done else "The next commit is not due yet.")
        return False
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Refusing to commit from a dirty checkout")
    state, queue = read_queue(root, queue_path)
    entry = queue[state["completed"] - 1]
    # Check every file before making any changes. Never overwrite unrelated edits.
    for name, expected in entry["before"].items():
        path = root / name
        actual = digest(path.read_bytes()) if path.exists() else None
        if actual != expected:
            raise RuntimeError(f"{name} changed outside the queue; manual review required")
    for name, content in entry["files"].items():
        (root / name).write_bytes(content.encode("utf-8"))
    state["completed"] = entry["number"]
    (root / STATE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    git(root, "add", "--", STATE, *entry["files"])
    git(root, "commit", "-m", entry["message"])
    # Never force-push. A concurrent remote change fails safely and can be retried.
    git(root, "push", "origin", "HEAD:main")
    print(f"Published commit {entry['number']}/10: {entry['message']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "bootstrap", "apply"])
    parser.add_argument("--queue", type=Path, default=Path(".automation-local/queue/queue.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "status":
        done, due = status(root)
        emit("done", str(done).lower())
        emit("due", str(due).lower())
    elif args.action == "bootstrap":
        payload = os.environ["QUEUE_PAYLOAD"]
        raw = gzip.decompress(base64.b64decode(payload, validate=True))
        args.queue.parent.mkdir(parents=True, exist_ok=True)
        args.queue.write_bytes(raw)
        read_queue(root, args.queue)
        print("Verified all nine queued changes; no commit was made.")
    else:
        apply_next(root, args.queue)


if __name__ == "__main__":
    main()
