"""sim/draft_eval.py match checks the first-pick table before its first game (harness only; live never loads it).

draft.choose_pick answers any error inside choose_pick_v2 with the old scorer's pick, so a DRAFT_FIRST_PICK_ADJUST path
that does not load used to hand an A/B arm's first two picks (the only ones a table touches) to the old scorer while the
run reported 0 errors. Now the table is loaded before any game is played or --out is opened: a path that does not load,
or a card the simulator's catalog does not hold, stops the run; otherwise its first line names the table and its count of
non-zero entries, so every arm's log records which table it ran."""

from __future__ import annotations

import json

import pytest

from agent.pokemon import draft
from sim import draft_eval, harness


class _NoEngine:
    def call(self, **msg):
        raise AssertionError("no engine call expected")

    def close(self) -> None:
        pass


@pytest.fixture
def match(monkeypatch, tmp_path):
    """Runs ``match`` with the shipped table and an empty cache, no engine, and every game answered by a stub;
    returns (run, games played, bridges opened, the --out path)."""
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV, raising=False)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})
    bridges: list[int] = []
    played: list[int] = []
    monkeypatch.setattr(harness, "Bridge", lambda: bridges.append(1) or _NoEngine())

    def fake_play(bridge, game_id, seed, specs, names, *, clock=False):
        played.append(seed)
        return {"winner": "A", "winner_side": "p1", "first_drafter": draft_eval.pool_of(seed)[1], "turns": 1, "clock": None,
                "rosters": {}, "teams": {}, "picks": []}

    monkeypatch.setattr(draft_eval, "play", fake_play)
    out = tmp_path / "arm.jsonl"

    def run() -> int:
        return draft_eval.main(["match", 'v2l{"lineup_sim_budget_s": 0}', "v1", "--pairs", "1", "--seed", "9", "--out", str(out)])

    return run, played, bridges, out


def _table(tmp_path, name: str, raw: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(raw))
    return str(path)


def test_a_mistyped_table_path_stops_the_run_before_any_game(match, monkeypatch, tmp_path):
    run, played, bridges, out = match
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, str(tmp_path / "first_pick_adjust_tf7.json"))  # the file is ..._tf07
    with pytest.raises(SystemExit, match="stopped before the first game: first-pick table .* unusable"):
        run()
    assert played == [] and bridges == [] and not out.exists()


def test_a_table_that_does_not_parse_stops_the_run_too(match, monkeypatch, tmp_path):
    run, played, bridges, out = match
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, _table(tmp_path, "bad.json", {"adjust_they_first": {"Archaludon": "plus a bit"}}))
    with pytest.raises(SystemExit, match="unusable"):
        run()
    assert played == [] and bridges == [] and not out.exists()


@pytest.mark.parametrize("raw, unknown", [
    ({"adjust_they_first": {"Archaludn": 0.7}}, "archaludn"),
    ({"pairs": {"Archaludon + Gyarrados": -0.5}}, "gyarrados"),
])
def test_a_card_the_catalog_does_not_hold_stops_the_run(match, monkeypatch, tmp_path, raw, unknown):
    # A misspelt card loads fine but can never act: the arm would quietly be the control.
    run, played, bridges, out = match
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, _table(tmp_path, "typo.json", raw))
    with pytest.raises(SystemExit, match=f"does not hold: \\['{unknown}'\\]"):
        run()
    assert played == [] and bridges == [] and not out.exists()


def test_the_table_in_use_is_the_first_line_of_the_log(match, monkeypatch, tmp_path, capsys):
    run, played, bridges, out = match
    assert run() == 0
    assert capsys.readouterr().out.splitlines()[0] == "first-pick table: shipped (0 non-zero entries)"
    assert len(played) == 2 and bridges == [1]
    path = _table(tmp_path, "tf.json", {"adjust_they_first": {"Archaludon": 0.7, "Gyarados": 0.0}, "pairs": {"Archaludon + Gyarados": -0.2}})
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, path)
    assert run() == 0
    assert capsys.readouterr().out.splitlines()[0] == f"first-pick table: {path} (2 non-zero entries)"
    assert len(played) == 4 and len(out.read_text().splitlines()) == 4
    assert draft.first_pick_adjust() is draft._ADJUST_TABLES[path]  # the players get the table the check loaded
