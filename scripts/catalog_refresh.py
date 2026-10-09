"""One step after every live session, before a deploy: the live catalog, tier entries and species weights for new cards.

    python scripts/catalog_refresh.py              # report and write
    python scripts/catalog_refresh.py --dry-run    # report only, nothing written

Plain language: the platform deals fixed sets we only learn by being offered them. After a session that showed a card
the catalog has not seen, four things must move together or the drafter misreads it and the tests fail: the catalog,
the card's tier entry (its base value in the draft), the species weight (Low Kick / Heavy Slam power) and the scenario
stand-ins. This script does the first three the way commits 53d4094 and 5d30dd0 did them by hand, and prints the fourth.

1. Pool (scripts/pool_from_live.py): every complete card in the captured draft states (tests/fixtures/live/*/) goes into
   data/cards.json and sim/cards.json. A capture wins over an older set under the same card id; a card the captures no
   longer hold is kept (a card seen once is known for good).
2. Tier entries for new cards only (no entry yet, or a set that no longer matches its entry's fingerprint): the matchup
   part by Monte Carlo over the whole catalog's board (scripts/build_tiers.matchup_parts: seed 1, 30,000 samples, the
   clamped draft weights), no card part (the fit never saw the card), zero adjustment; every other rating untouched.
   data/card_tiers.json and docs/card_tiers.md (the docs table is rebuilt from the JSON, so it never drifts).
3. Species weights: SPECIES_WEIGHT_KG in agent/pokemon/data.py gains every catalog species it lacks, from the vendored
   dex (data/pokedex.json, built from the Showdown dex).
4. Prints which hand-written stand-ins in scripts/scenarios.py (HAND_SETS) are live cards now and should move to
   LIVE_SPECIES. That file is the eval lane's: this script never edits it.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_tiers  # noqa: E402
import pool_from_live  # noqa: E402

from agent.pokemon import data, matchup  # noqa: E402

CARDS_FILES = ("data/cards.json", "sim/cards.json")
TIERS_FILE = "data/card_tiers.json"
TIERS_DOC = "docs/card_tiers.md"
DATA_PY = "agent/pokemon/data.py"
SCENARIOS = "scripts/scenarios.py"
LOCAL_TZ = "America/Los_Angeles"  # match days are written in Pacific time ("first seen live Oct 8")
TABLE_HEAD = ("| Tier | Card | Rating | Sim | Why | Flag / live |", "|---|---|---:|---:|---|---|")
MATCHUP_ONLY = re.compile(r"\s*(?P<names>[^.]*?) \(first seen live (?P<when>[^)]*)\) carry their matchup part only")


def _day(stamp: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo

        stamp = stamp.astimezone(ZoneInfo(LOCAL_TZ))
    except Exception:  # noqa: BLE001 — no tz database: the UTC day is close enough for a note
        pass
    return f"{stamp:%b} {stamp.day}"


def first_seen(root: Path) -> dict[str, str]:
    """card_id -> the Pacific day it was first offered in a captured draft (its earliest decision deadline)."""
    seen: dict[str, datetime] = {}
    for path in sorted(root.glob("tests/fixtures/live/*/draft-*.json")):
        try:
            obs = json.loads(path.read_text(encoding="utf-8")).get("observation") or {}
            when = datetime.fromisoformat(str(obs.get("decision_deadline_at")))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        for card in obs.get("available_cards") or []:
            cid = card.get("card_id") if isinstance(card, dict) else None
            if cid and (cid not in seen or when < seen[cid]):
                seen[cid] = when
    return {cid: _day(when) for cid, when in seen.items()}


def merge_catalog(current: list[dict], live: dict[str, dict]) -> tuple[list[dict], list[str], list[str], list[str]]:
    """(the merged catalog sorted by species, new card ids, changed card ids, ids kept that no capture holds)."""
    fields = pool_from_live.FIELDS
    by_id = {c["card_id"]: c for c in current}
    new = sorted(cid for cid in live if cid not in by_id)
    changed = sorted(cid for cid in live if cid in by_id and {f: by_id[cid].get(f) for f in fields} != {f: live[cid].get(f) for f in fields})
    kept = sorted(cid for cid in by_id if cid not in live)
    merged = {**by_id, **live}
    return sorted(merged.values(), key=lambda c: c["species"]), new, changed, kept


def tier_entries(cards: list[dict], tiers: dict[str, dict], fresh: list[str], model_path: Path, days: dict[str, str],
                 today: str) -> dict[str, dict]:
    """card_id -> tier entry for each card in ``fresh``, built the way build_tiers.main builds one, without the card
    part or an adjustment; the why line ranks the other cards by their stored ratings."""
    units = [matchup.unit(c) for c in cards]
    board = matchup.Board(units, matchup.load_model(model_path, section="draft"), tiers={})
    parts = build_tiers.matchup_parts(board, random.Random(1))
    index = {c["card_id"]: k for k, c in enumerate(cards)}
    ratings = [float((tiers.get(c["card_id"]) or {}).get("rating", 100 * parts[k])) for k, c in enumerate(cards)]
    for cid in fresh:
        ratings[index[cid]] = round(100 * parts[index[cid]], 1)
    out = {}
    for cid in fresh:
        k = index[cid]
        unit, rating = units[k], ratings[k]
        seen = f"not in the fitted model (first seen live {days.get(cid, today)})"
        prior = build_tiers.PRIORS.get(unit.species, {}).get("flag")
        out[cid] = {"species": unit.species, "fingerprint": unit.key, "tier": build_tiers.tier_of(rating), "rating": rating,
                    "sim_rating": rating, "adjust": 0, "matchup_part": rating, "card_part": 0.0,
                    "why": f"{build_tiers.why(board, k, 0.0, ratings)}; {seen}: matchup part only",
                    "flag": seen + (f"; {prior}" if prior else "")}
    return out


def _with_names(text: str, names: list[str], days: list[str]) -> str:
    """``text`` with ``names`` added to its 'X, Y and Z (first seen live D) carry their matchup part only' sentence."""
    found = MATCHUP_ONLY.search(text)
    listed = [n.strip() for n in found.group("names").replace(" and ", ", ").split(", ")] if found else []
    when = [w.strip() for w in found.group("when").split(" / ")] if found else []
    listed += [n for n in names if n not in listed]
    when += [d for d in days if d not in when]
    joined = listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + " and " + listed[-1]
    sentence = f"{joined} (first seen live {' / '.join(when)}) carry their matchup part only"
    if found:
        return text[:found.start("names")] + sentence + text[found.end():]
    return text.rstrip(".") + f". {sentence}."


def tiers_text(blob: dict, entries: dict[str, dict], names: list[str], days: list[str]) -> str:
    cards = dict(blob["cards"])
    cards.update(entries)
    ordered = dict(sorted(cards.items(), key=lambda kv: -kv[1]["rating"]))  # stable: the file is already in this order
    note = _with_names(blob["note"], names, days) if names else blob["note"]
    return json.dumps({**blob, "note": note, "cards": ordered}, indent=1, ensure_ascii=False) + "\n"


def doc_text(doc: str, blob: dict, names: list[str], days: list[str]) -> str:
    """docs/card_tiers.md: its header with the note's names, then the table rebuilt from the tier JSON (the format
    build_tiers.main writes)."""
    lines = doc.splitlines()
    head = lines[:lines.index(TABLE_HEAD[0])] if TABLE_HEAD[0] in lines else lines
    if names:
        head = [_with_names(line, names, days) if "Rebuild after refitting" in line else line for line in head]
    rows = []
    for entry in blob["cards"].values():
        note = "; ".join(x for x in (entry.get("flag"), entry.get("live")) if x)
        rows.append(f"| {entry['tier']} | {entry['species']} | {entry['rating']:+.0f} | {entry['sim_rating']:+.0f} | {entry['why']} | {note} |")
    return "\n".join(head + list(TABLE_HEAD) + rows) + "\n"


def missing_weights(root: Path, species: list[str]) -> tuple[dict[str, float], list[str], tuple[int, int] | None]:
    """({species id: kg from data/pokedex.json} for catalog species SPECIES_WEIGHT_KG lacks, species the dex has no
    weight for, (first, last) line of the dict literal in agent/pokemon/data.py)."""
    source = (root / DATA_PY).read_text(encoding="utf-8")
    table, span = set(), None
    for node in ast.parse(source).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if getattr(target, "id", None) == "SPECIES_WEIGHT_KG" and isinstance(node.value, ast.Dict):
            table = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            span = (node.value.lineno, node.value.end_lineno)
    dex = json.loads((root / "data" / "pokedex.json").read_text(encoding="utf-8"))
    add, unknown = {}, []
    for name in species:
        key = data.to_id(name)
        if key in table or key in add:
            continue
        kg = (dex.get(key) or {}).get("weightkg")
        if kg:
            add[key] = float(kg)
        else:
            unknown.append(name)
    return add, unknown, span


def weights_text(root: Path, add: dict[str, float], span: tuple[int, int]) -> str:
    """agent/pokemon/data.py with one line of new weights before the table's closing brace."""
    lines = (root / DATA_PY).read_text(encoding="utf-8").splitlines(keepends=True)
    entry = "    " + " ".join(f'"{k}": {v!r},' for k, v in add.items()) + "\n"
    end = span[1] - 1  # 0-based index of the line holding the closing brace
    return "".join(lines[:end] + [entry] + lines[end:])


def scenario_stand_ins(root: Path, catalog_species: set[str]) -> list[str]:
    """Species scripts/scenarios.py still writes by hand (HAND_SETS) that the catalog holds now."""
    path = root / SCENARIOS
    if not path.exists():
        return []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "HAND_SETS" and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            return [s for s in sorted(catalog_species) if data.to_id(s) in keys]
    return []


def refresh(root: Path = ROOT, *, dry_run: bool = False, today: str | None = None, say=print) -> dict:
    """Run the four steps on the checkout at ``root``; returns what changed (paths written, new ids, ratings...)."""
    today = today or _day(datetime.now().astimezone())
    current = json.loads((root / CARDS_FILES[0]).read_text(encoding="utf-8"))
    cards, new, changed, kept = merge_catalog(current, pool_from_live.live_cards(root))
    blob = json.loads((root / TIERS_FILE).read_text(encoding="utf-8"))
    entries_by_id = blob.get("cards") or {}
    fresh = [c["card_id"] for c in cards
             if (entries_by_id.get(c["card_id"]) or {}).get("fingerprint") != matchup.fingerprint(c)]
    days = first_seen(root)
    entries = tier_entries(cards, entries_by_id, fresh, root / "data" / "draft_model.json", days, today) if fresh else {}
    names = [entries[cid]["species"] for cid in fresh]
    when = list(dict.fromkeys(days.get(cid, today) for cid in fresh))
    species = {c["species"] for c in cards}
    add, unknown, span = missing_weights(root, sorted(species))
    stand_ins = scenario_stand_ins(root, species)

    species_of = {c["card_id"]: c["species"] for c in cards}
    say(f"catalog: {len(cards)} cards; new {[species_of[c] for c in new]}; changed sets {[species_of[c] for c in changed]}; "
        f"kept though no capture holds them {[species_of[c] for c in kept]}")
    for cid in fresh:
        e = entries[cid]
        say(f"tier entry: {e['species']} {e['rating']:+.1f} ({e['tier']}): {e['why']}")
    if add:
        say("species weights from data/pokedex.json: " + ", ".join(f"{k} {v} kg" for k, v in add.items()))
    if unknown:
        say(f"WARNING no weight in data/pokedex.json for {unknown}: rebuild the dex (scripts/build_dex.py) and rerun")
    if stand_ins:
        say(f"scripts/scenarios.py (eval lane, not edited here): move {stand_ins} from HAND_SETS to LIVE_SPECIES")
    for cid in fresh:
        card = next(c for c in cards if c["card_id"] == cid)
        say(f"review the flag of {species_of[cid]}: ability {card.get('ability')}, item {card.get('item')}; say what the "
            f"draft value does not model next to 'not in the fitted model'")

    writes: dict[str, str] = {}
    cards_text = json.dumps(cards, indent=1, ensure_ascii=False) + "\n"
    for rel in CARDS_FILES:
        path = root / rel
        if not path.exists() or path.read_text(encoding="utf-8") != cards_text:
            writes[rel] = cards_text
    if entries:
        writes[TIERS_FILE] = tiers_text(blob, entries, names, when)
        new_blob = json.loads(writes[TIERS_FILE])
        writes[TIERS_DOC] = doc_text((root / TIERS_DOC).read_text(encoding="utf-8"), new_blob, names, when)
    if add and span:
        writes[DATA_PY] = weights_text(root, add, span)
    if dry_run:
        say("dry run: would write " + (", ".join(writes) if writes else "nothing"))
    else:
        for rel, text in writes.items():
            (root / rel).write_text(text, encoding="utf-8")
        say("wrote " + (", ".join(writes) if writes else "nothing: the catalog, tiers and weights are up to date"))
    return {"new": new, "changed": changed, "kept": kept, "fresh": fresh, "entries": entries, "weights": add,
            "unknown_weights": unknown, "stand_ins": stand_ins, "writes": sorted(writes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout to refresh (default: this one)")
    args = parser.parse_args(argv)
    refresh(args.root.resolve(), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
