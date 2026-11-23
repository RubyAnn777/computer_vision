import hashlib
import json
from pathlib import Path

from computer_vision.detection.run_info import collect_run_info, git_state, sha256_of


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"product")
    assert sha256_of(path) == hashlib.sha256(b"product").hexdigest()


def test_git_state_outside_a_repo_is_marked_unknown_and_dirty(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"commit": "unknown", "dirty": True}


def test_collect_run_info_is_json_serialisable(tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    path.write_text("page_id,split\n")
    info = collect_run_info(tmp_path, {"split": path}, {"settings": {"imgsz": 1280}})

    assert info["inputs_sha256"] == {"split": sha256_of(path)}
    assert info["settings"] == {"imgsz": 1280}
    assert "git" in info
    json.dumps(info)
