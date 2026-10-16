import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.commit_queue import STATE, apply_next, digest, git, read_queue, status

ROOT = Path(__file__).resolve().parents[1]


class CommitQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo = base / "work"
        remote = base / "origin.git"
        self.repo.mkdir()
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "Pipeline Test")
        git(self.repo, "config", "user.email", "pipeline-test@example.invalid")
        git(self.repo, "config", "commit.gpgsign", "false")
        git(self.repo, "config", "core.autocrlf", "false")
        git(self.repo, "remote", "add", "origin", str(remote))
        (self.repo / ".github").mkdir()
        self.queue_path = base / "queue.json"
        queue = []
        previous = None
        for number in range(2, 11):
            content = f"STAGE = {number}\n"
            queue.append({"number": number, "message": f"Step {number}",
                          "before": {"SqueezeNet_EuroSAT.py": previous},
                          "files": {"SqueezeNet_EuroSAT.py": content}})
            previous = digest(content.encode())
        self.write_queue(queue)
        git(self.repo, "add", STATE)
        git(self.repo, "commit", "-m", "Initial setup")
        git(self.repo, "push", "-u", "origin", "main")

    def write_queue(self, queue):
        raw = json.dumps(queue).encode()
        self.queue_path.write_bytes(raw)
        (self.repo / STATE).write_text(json.dumps({"completed": 1, "total": 10,
                                                   "interval_hours": 2,
                                                   "queue_sha256": digest(raw)}))

    def due(self):
        return int(git(self.repo, "log", "-1", "--format=%ct", "--", STATE)) + 7200

    def apply(self, now=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return apply_next(self.repo, self.queue_path, self.due() if now is None else now)

    def test_two_hour_gate_and_no_commit_on_early_retry(self):
        self.assertFalse(self.apply(self.due() - 1))
        self.assertEqual(git(self.repo, "rev-list", "--count", "HEAD"), "1")
        self.assertTrue(self.apply())
        self.assertFalse(self.apply(self.due() - 1))
        self.assertEqual(git(self.repo, "rev-list", "--count", "HEAD"), "2")

    def test_exactly_ten_commits_and_safe_completed_retry(self):
        for completed in range(2, 11):
            self.assertTrue(self.apply())
            self.assertEqual(json.loads((self.repo / STATE).read_text())["completed"], completed)
            self.assertEqual(git(self.repo, "rev-list", "--count", "HEAD"), str(completed))
        self.assertEqual(status(self.repo, self.due()), (True, False))
        self.assertFalse(self.apply())
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"),
                         git(self.repo, "rev-parse", "origin/main"))

    def test_queue_tampering_is_rejected(self):
        self.queue_path.write_bytes(self.queue_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.apply()

    def test_unrelated_untracked_file_is_not_committed(self):
        (self.repo / "personal.txt").write_text("unrelated")
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            self.apply()

    def test_existing_committed_changes_are_not_overwritten(self):
        (self.repo / "SqueezeNet_EuroSAT.py").write_text("# manual change\n")
        git(self.repo, "add", "SqueezeNet_EuroSAT.py")
        git(self.repo, "commit", "-m", "Manual work")
        with self.assertRaisesRegex(RuntimeError, "manual review"):
            self.apply()

    def test_unexpected_paths_rejected_even_with_matching_queue_hash(self):
        queue = json.loads(self.queue_path.read_bytes())
        queue[0]["files"] = {"../escape.py": "pass\n"}
        queue[0]["before"] = {"../escape.py": None}
        self.write_queue(queue)
        with self.assertRaisesRegex(ValueError, "unexpected path"):
            read_queue(self.repo, self.queue_path)

    @unittest.skipUnless(
        (ROOT / ".automation-local/queue.json").exists()
        and json.loads((ROOT / STATE).read_text())["completed"] == 1,
        "Full replay uses the local queue and initial-commit README",
    )
    def test_full_notebook_queue_replays_to_the_reviewed_preview(self):
        queue_path = ROOT / ".automation-local/queue.json"
        self.queue_path.write_bytes(queue_path.read_bytes())
        (self.repo / STATE).write_bytes((ROOT / STATE).read_bytes())
        (self.repo / "README.md").write_bytes((ROOT / "README.md").read_bytes())
        git(self.repo, "add", "README.md", STATE)
        git(self.repo, "commit", "--amend", "--no-edit")
        # Reset only the disposable test origin to this revised initial fixture.
        git(self.repo, "push", "--force", "origin", "main")
        for _ in range(9):
            self.assertTrue(self.apply())
        self.assertEqual(git(self.repo, "rev-list", "--count", "HEAD"), "10")
        self.assertEqual((self.repo / "SqueezeNet_EuroSAT.py").read_bytes(),
                         (ROOT / ".automation-local/SqueezeNet_EuroSAT.preview.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
