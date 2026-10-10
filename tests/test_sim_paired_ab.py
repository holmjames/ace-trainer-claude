"""sim/paired_ab.py: the read-out of a paired draft A/B (two sim/draft_eval.py match --out files on one --seed).

Games pair by (battle seed, our seat), never by row order: match writes no row for a game that raised, so one error in
one arm used to shift every later pair of a row-order read-out. A file holding a game twice stops the read-out. With
--not-held SPECIES the primary endpoint of a they-first table A/B is fixed by the control's draft alone: they-first games
where the opponent did not open with SPECIES and the control's first two picks did not take it."""

from __future__ import annotations

import json
import math
import random

import pytest

from sim import paired_ab

BOT = "checkout:/opponents/search-bot:checkout_shim:/venv/bin/python"
OURS = 'v2l{"lineup_sim_budget_s": 0}'
FIRST_SLOTS = (0, 3, 4, 7, 8, 11)  # the first drafter's picks in the snake (sim/harness.py)
MINE = ("Iron Hands", "Gholdengo", "Rillaboom", "Sylveon", "Kingambit", "Amoonguss")
THEIRS = ("Incineroar", "Gyarados", "Porygon2", "Dragonite", "Tornadus", "Whimsicott")


def row(seed: int, us: str, won: bool | None, *, we_first: bool = True, ours=MINE, theirs=THEIRS, spec: str = OURS) -> dict:
    them = "p2" if us == "p1" else "p1"
    first = us if we_first else them
    mine, other = iter(ours), iter(theirs)
    picks = []
    for i in range(12):
        side = first if i in FIRST_SLOTS else (them if first == us else us)
        picks.append({"side": side, "species": next(mine) if side == us else next(other), "public_reason": ""})
    winner = None if won is None else (us if won else them)
    return {"seed": seed, "specs": {us: spec, them: BOT}, "first": first, "winner": winner, "turns": 5, "picks": picks}


def write(path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def row_order_z(control: list[dict], arm: list[dict]) -> float:
    """What a row-order read-out computes: rows zipped in file order, both cut to the shorter file."""
    def won(r):
        return r["winner"] == next(k for k, v in r["specs"].items() if "checkout_shim" not in v)

    pairs = list(zip(map(won, control), map(won, arm)))
    a_only = sum(1 for c, a in pairs if a and not c)
    c_only = sum(1 for c, a in pairs if c and not a)
    return (a_only - c_only) / math.sqrt(a_only + c_only) if a_only + c_only else 0.0


def test_our_seat_is_the_one_seat_that_is_not_the_shimmed_clone():
    assert paired_ab.our_seat({"p1": BOT, "p2": OURS}) == "p2"
    assert paired_ab.our_seat({"p1": OURS, "p2": BOT}) == "p1"
    for specs in ({"p1": OURS, "p2": "v1"}, {"p1": BOT, "p2": BOT}):
        with pytest.raises(ValueError, match="unclear which one is ours"):
            paired_ab.our_seat(specs)


def test_an_error_in_one_arm_leaves_one_game_unpaired_instead_of_shifting_the_rest(tmp_path, capsys):
    # An A/A: both arms won exactly the same games, but the arm's game (2, p2) raised, so match wrote no row for it.
    games = [(seed, us, seed <= 2) for seed in (1, 2, 3, 4) for us in ("p1", "p2")]
    control = [row(seed, us, won, we_first=seed % 2 == 0) for seed, us, won in games]
    arm = [row(seed, us, won, we_first=seed % 2 == 0) for seed, us, won in games if (seed, us) != (2, "p2")]
    assert row_order_z(control, arm) == -1.0  # row order pairs (2, p2) with (3, p1): a discordant pair that is not a pair
    c, a = paired_ab.load(write(tmp_path / "c.jsonl", control)), paired_ab.load(write(tmp_path / "a.jsonl", arm[::-1]))
    keys = sorted(set(c) & set(a))
    assert len(keys) == 7 and sorted(set(c) - set(a)) == [(2, "p2")]
    s = paired_ab.paired(keys, c, a)
    assert (s["n"], s["arm"], s["control"], s["arm_only"], s["control_only"], s["z"]) == (7, 3, 3, 0, 0, 0.0)
    assert paired_ab.main([str(tmp_path / "c.jsonl"), str(tmp_path / "a.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "games paired by (battle seed, our seat): 7; only in control: 1; only in arm: 0" in out
    assert "WARNING: games in one file only" in out and "[(2, 'p2')]" in out
    assert "discordant arm-only 0, control-only 0; McNemar z = +0.00" in out


def test_complete_files_in_the_same_order_give_the_row_order_numbers(tmp_path, capsys):
    rng = random.Random(5)
    keys = [(seed, us) for seed in range(100, 140) for us in ("p1", "p2")]
    control = [row(seed, us, rng.random() < 0.5, we_first=(seed + (us == "p2")) % 2 == 0) for seed, us in keys]
    arm = [row(seed, us, rng.random() < 0.55, we_first=(seed + (us == "p2")) % 2 == 0) for seed, us in keys]
    c, a = paired_ab.load(write(tmp_path / "c.jsonl", control)), paired_ab.load(write(tmp_path / "a.jsonl", arm))
    s = paired_ab.paired(sorted(c), c, a)
    assert s["n"] == 80 and s["z"] == pytest.approx(row_order_z(control, arm))
    assert s["arm_only"] + s["control_only"] > 0
    assert paired_ab.main([str(tmp_path / "c.jsonl"), str(tmp_path / "a.jsonl")]) == 0
    assert f"McNemar z = {s['z']:+.2f}" in capsys.readouterr().out


def test_a_game_twice_in_one_file_stops_the_read_out(tmp_path):
    rows = [row(1, "p1", True), row(1, "p2", False)]
    path = write(tmp_path / "twice.jsonl", rows + rows[:1])  # --out appends: a second run into the same file
    with pytest.raises(SystemExit, match="1 games appear twice .* give every run a fresh file"):
        paired_ab.load(path)


def test_files_that_are_not_the_same_games_stop_the_read_out(tmp_path):
    c = write(tmp_path / "c.jsonl", [row(1, "p1", True), row(1, "p2", True)])
    with pytest.raises(SystemExit, match="same file"):
        paired_ab.main([c, c])
    with pytest.raises(SystemExit, match="no game is in both files"):
        paired_ab.main([c, write(tmp_path / "other-seed.jsonl", [row(2, "p1", True)])])
    with pytest.raises(SystemExit, match="different first drafter"):
        paired_ab.main([c, write(tmp_path / "other-order.jsonl", [row(1, "p1", True, we_first=False)])])


def test_not_held_is_fixed_by_the_controls_draft_alone(tmp_path, capsys):
    arch = ("Archaludon", *MINE[1:])
    second = ("Iron Hands", "Archaludon", *MINE[2:])
    opened = ("Archaludon", *THEIRS[1:])
    control = [
        row(1, "p1", True, we_first=False, theirs=opened),  # the opponent opened with it: no table can act
        row(2, "p1", True, we_first=False, ours=arch),  # the control took it at pick 2
        row(3, "p1", False, we_first=False, ours=second),  # ... at pick 3
        row(4, "p1", False, we_first=False),  # primary: the arm takes it and wins
        row(5, "p1", False, we_first=False),  # primary: the arm takes it and wins
        row(6, "p1", True, we_first=False),  # primary: the arm does not take it and loses
        row(7, "p1", True, we_first=True),  # we drafted first: never in the subset
    ]
    arm = [
        row(1, "p1", False, we_first=False, theirs=opened),
        row(2, "p1", True, we_first=False, ours=arch),
        row(3, "p1", True, we_first=False, ours=second),
        row(4, "p1", True, we_first=False, ours=arch),
        row(5, "p1", True, we_first=False, ours=second),
        row(6, "p1", False, we_first=False),
        row(7, "p1", True, we_first=True),
    ]
    cp, ap = write(tmp_path / "c8.jsonl", control), write(tmp_path / "tf.jsonl", arm)
    c, a = paired_ab.load(cp), paired_ab.load(ap)
    keys = sorted(set(c) & set(a))
    primary, opened_keys, took = paired_ab.not_held(keys, c, "archaludon")
    assert (primary, opened_keys, took) == ([(4, "p1"), (5, "p1"), (6, "p1")], [(1, "p1")], [(2, "p1"), (3, "p1")])
    flipped = {k: {**g, "won": not g["won"], "ours": list(MINE)} for k, g in a.items()}  # the arm's games never move it
    assert paired_ab.not_held(keys, c, "Archaludon") == paired_ab.not_held(sorted(set(c) & set(flipped)), c, "Archaludon")
    s = paired_ab.paired(primary, c, a)
    assert (s["arm"], s["control"], s["arm_only"], s["control_only"]) == (2, 1, 2, 1)
    assert paired_ab.main([cp, ap, "--not-held", "Archaludon"]) == 0
    out = capsys.readouterr().out
    assert "primary endpoint (fixed by the control's draft): the 3 of 6 they-first paired games" in out
    assert "the arm took Archaludon in its first two picks in 2 of them; the opponent opened with it in the arm's game in 0" in out
    assert "set apart, the opponent opened with Archaludon (no table can act): 1 games" in out
    assert "set apart, the control took Archaludon at picks 2-3: 2 games" in out


def test_a_misspelt_not_held_card_stops_before_any_number(tmp_path, capsys):
    c = write(tmp_path / "c.jsonl", [row(1, "p1", True, we_first=False)])
    a = write(tmp_path / "a.jsonl", [row(1, "p1", False, we_first=False)])
    with pytest.raises(SystemExit, match="no draft in either file picked it"):
        paired_ab.main([c, a, "--not-held", "Archaludn"])
    assert capsys.readouterr().out == ""
