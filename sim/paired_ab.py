#!/usr/bin/env python3
"""Paired read-out of a draft A/B: two ``sim/draft_eval.py match --out`` files run with the same --seed, --pairs and flags.

    python3 -I sim/paired_ab.py CONTROL.jsonl ARM.jsonl [--not-held SPECIES]

Games are paired by key, never by row order: (battle seed, our seat), "ours" being the one seat whose spec is not a
checkout_shim clone (the search bot). ``match`` writes no row for a game that raised, so pairing by row order shifts every
later pair after one error in one arm; here that game is simply unpaired, and the unpaired games of each arm are counted
and left out of every number. A file that holds a key twice (``--out`` appends, so two runs into one file) stops the
read-out, as do files that share no game or disagree on a paired game's first drafter.

Prints, on the paired games: each arm's wins, the discordant pairs, McNemar z = (arm-only - control-only) / sqrt(discordant)
(a row-order read-out's z whenever both files hold the same games in the same order) and the exact two-sided sign test;
the same by draft order; and how many paired games drafted differently (our picks; our four and leads where our picks
match, which is the whole of an A/A check on our side). A win is our seat winning; an unfinished game (turn cap) counts
as not won.

``--not-held SPECIES`` adds a they-first table's primary endpoint, fixed by the CONTROL's draft before any result is read:
the they-first games in which the opponent did not open with SPECIES and the control's first two picks (overall picks 2
and 3) did not take it, i.e. the games where an ``adjust_they_first`` entry for SPECIES can act. Reported like the rest,
with how often the arm took SPECIES in its first two picks there. The they-first games the opponent opened with SPECIES
(no table can act) and those the control took it in are reported apart. Pass the control first: it fixes the subset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

SHIM = "checkout_shim"


def to_id(name: object) -> str:
    """A species id, as agent/pokemon/data.py ``to_id``: lowercase letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def our_seat(specs: dict) -> str:
    """The one seat whose spec is not a checkout_shim clone."""
    seats = [side for side, spec in specs.items() if SHIM not in str(spec)]
    if len(seats) != 1:
        raise ValueError(f"{len(seats)} seats are not a {SHIM} clone, so it is unclear which one is ours")
    return seats[0]


def load(path: str) -> dict[tuple, dict]:
    """{(battle seed, our seat): game} for one ``match --out`` file."""
    games: dict[tuple, dict] = {}
    twice: list[tuple] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                us = our_seat(row["specs"])
            except ValueError as exc:
                raise SystemExit(f"{path}:{n}: {exc}") from exc
            key = (row["seed"], us)
            if key in games:
                twice.append(key)
                continue
            picks = row.get("picks") or []
            theirs = [p["species"] for p in picks if p["side"] != us]
            games[key] = {"won": row["winner"] == us, "unfinished": row["winner"] is None, "we_first": row["first"] == us,
                          "ours": [p["species"] for p in picks if p["side"] == us], "their_first": theirs[0] if theirs else None,
                          "draft": [(p["side"], p["species"]) for p in picks], "four": (row.get("teams") or {}).get(us, []),
                          "spec": row["specs"][us],
                          "opponent": next(spec for side, spec in row["specs"].items() if side != us)}
    if twice:
        raise SystemExit(f"{path}: {len(twice)} games appear twice (same battle seed and seat), e.g. {twice[:3]}: "
                         "--out appends, so give every run a fresh file")
    return games


def sign_test(a: int, b: int) -> float:
    """Exact two-sided binomial test of a vs b discordant results (p = 0.5)."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(keys: list[tuple], control: dict, arm: dict) -> dict:
    """Each arm's wins on ``keys``, the discordant pairs, McNemar z (arm minus control) and the sign test."""
    a_only = sum(1 for k in keys if arm[k]["won"] and not control[k]["won"])
    c_only = sum(1 for k in keys if control[k]["won"] and not arm[k]["won"])
    d = a_only + c_only
    return {"n": len(keys), "control": sum(control[k]["won"] for k in keys), "arm": sum(arm[k]["won"] for k in keys),
            "arm_only": a_only, "control_only": c_only, "z": (a_only - c_only) / math.sqrt(d) if d else 0.0,
            "p": sign_test(a_only, c_only)}


def describe(s: dict) -> str:
    n = s["n"]
    return (f"arm {s['arm']}/{n} vs control {s['control']}/{n} ({100 * (s['arm'] - s['control']) / max(1, n):+.1f} points); "
            f"discordant arm-only {s['arm_only']}, control-only {s['control_only']}; McNemar z = {s['z']:+.2f}; "
            f"sign test p = {s['p']:.3f}")


def not_held(keys: list[tuple], control: dict, species: str) -> tuple[list, list, list]:
    """The they-first paired games split by the CONTROL's draft alone: (primary: the opponent did not open with ``species``
    and the control's first two picks did not take it; the opponent opened with it; the control took it)."""
    want = to_id(species)
    they_first = [k for k in keys if not control[k]["we_first"]]
    opened = [k for k in they_first if to_id(control[k]["their_first"]) == want]
    took = [k for k in they_first if k not in opened and want in {to_id(s) for s in control[k]["ours"][:2]}]
    primary = [k for k in they_first if k not in opened and k not in took]
    return primary, opened, took


def spec_of(games: dict) -> str:
    specs = sorted({g["spec"] for g in games.values()})
    return specs[0] if len(specs) == 1 else f"{len(specs)} different specs: {specs}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control", help="the control arm's match --out file (it fixes the --not-held subset)")
    ap.add_argument("arm", help="the treated arm's match --out file, run with the same --seed, --pairs and flags")
    ap.add_argument("--not-held", metavar="SPECIES", default=None,
                    help="also report the they-first games where neither the opponent's first pick nor the control's first two took it")
    args = ap.parse_args(argv)
    if os.path.realpath(args.control) == os.path.realpath(args.arm):
        raise SystemExit("the control and the arm are the same file")
    control, arm = load(args.control), load(args.arm)
    species = args.not_held
    if species and to_id(species) not in {to_id(s) for g in (*control.values(), *arm.values()) for _, s in g["draft"]}:
        raise SystemExit(f"--not-held {species!r}: no draft in either file picked it (spelling?)")
    keys = sorted(set(control) & set(arm))
    only_c, only_a = sorted(set(control) - set(arm)), sorted(set(arm) - set(control))
    print(f"control {os.path.basename(args.control)}: {len(control)} games; ours {spec_of(control)}")
    print(f"arm     {os.path.basename(args.arm)}: {len(arm)} games; ours {spec_of(arm)}")
    if spec_of(control) == spec_of(arm):
        print("  both arms list the same spec for us, so they differ only by environment: each arm's match log names its "
              "first-pick table on its first line")
    if not keys:
        raise SystemExit("no game is in both files: were they run with the same --seed, --pairs, --pool-has and --first-only?")
    order = [k for k in keys if control[k]["we_first"] != arm[k]["we_first"]]
    if order:
        raise SystemExit(f"{len(order)} paired games have a different first drafter in the two files, e.g. {order[:3]}: not the same pools")
    print(f"games paired by (battle seed, our seat): {len(keys)}; only in control: {len(only_c)}; only in arm: {len(only_a)}")
    if only_c or only_a:
        print(f"  WARNING: games in one file only (an error in one arm, a run cut short or other flags), left out below: "
              f"{len(only_c) + len(only_a)}; control-only e.g. {only_c[:3]}, arm-only e.g. {only_a[:3]}")
    other = sum(1 for k in keys if control[k]["opponent"] != arm[k]["opponent"])
    if other:
        print(f"  WARNING: in {other} paired games the arms played different opponent specs")
    print(f"all paired games: {describe(paired(keys, control, arm))}")
    unfinished = (sum(control[k]["unfinished"] for k in keys), sum(arm[k]["unfinished"] for k in keys))
    if any(unfinished):
        print(f"  unfinished (turn cap; counted as not won): control {unfinished[0]}, arm {unfinished[1]}")
    for we_first, label in ((True, "we first  "), (False, "they first")):
        subset = [k for k in keys if control[k]["we_first"] == we_first]
        if subset:
            print(f"  {label}: {describe(paired(subset, control, arm))}")
    same_ours = [k for k in keys if control[k]["ours"] == arm[k]["ours"]]
    first_two = sum(1 for k in keys if control[k]["ours"][:2] != arm[k]["ours"][:2])
    four = sum(1 for k in same_ours if control[k]["four"] != arm[k]["four"])
    same = [k for k in keys if control[k]["draft"] == arm[k]["draft"]]
    agree = sum(1 for k in same if control[k]["won"] == arm[k]["won"])
    print(f"our picks differ in {len(keys) - len(same_ours)} of {len(keys)} paired games (the first two in {first_two}); where "
          f"they are the same, our four and leads differ in {four}; identical drafts (all twelve picks): {len(same)}, "
          f"same result in {agree}")
    if species:
        primary, opened, took = not_held(keys, control, species)
        they_first = len(primary) + len(opened) + len(took)
        print(f"--not-held {species}, primary endpoint (fixed by the control's draft): the {len(primary)} of {they_first} they-first "
              f"paired games where the opponent did not open with {species} and the control's first two picks did not take it")
        if primary:
            print(f"  {describe(paired(primary, control, arm))}")
            arm_took = sum(1 for k in primary if to_id(species) in {to_id(s) for s in arm[k]["ours"][:2]})
            moved = sum(1 for k in primary if to_id(arm[k]["their_first"]) == to_id(species))
            print(f"  the arm took {species} in its first two picks in {arm_took} of them; the opponent opened with it in the "
                  f"arm's game in {moved}")
        for label, subset in ((f"the opponent opened with {species} (no table can act)", opened),
                              (f"the control took {species} at picks 2-3", took)):
            if subset:
                print(f"  set apart, {label}: {len(subset)} games; {describe(paired(subset, control, arm))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
