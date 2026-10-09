"""The code's own prediction of what the opponent does this turn (A4): one line per opposing slot, in the vocabulary
the reply mixture uses (attack / protect / switch / setup / fakeout), from the opponent class the memory infers and
the priors the core lane measured live (agent/pokemon/replies.py, single source). The model answers with its own
prediction in the same vocabulary; an override that trails the best row is accepted only when the model's prediction
matches this top column (the review's E2). When the reply mixture ran this turn, its own top column is used."""

from __future__ import annotations

from typing import Any

from . import battle, data
from .memory import species_key

KINDS = ("attack", "protect", "switch", "setup", "fakeout")
SETUP_MOVES = {"nastyplot", "calmmind", "swordsdance", "dragondance", "bulkup", "quiverdance", "tailwind", "trickroom", "icywind",
               "electroweb", "snarl", "lightscreen", "reflect"}


def _priors():
    """Core's measured priors when importable; the brief's numbers otherwise."""
    try:
        from .replies import _p_fakeout, _p_protect, _p_setup, _p_switch

        return _p_protect, _p_switch, _p_setup, _p_fakeout
    except Exception:  # noqa: BLE001 — a missing or changed module must not stop a turn
        def p_protect(klass, *, streak, ko_one, ko_two, facing, fresh_in, turn):
            if klass == "random":
                return 0.15
            if klass == "llm":
                return (0.5 if ko_one >= 0.5 else 0.3 if ko_two >= 0.5 else 0.1) * (0.5 if streak else 1.0)
            if streak:
                return 0.0
            return 0.6 if ko_one >= 0.5 else 0.4 if ko_two >= 0.5 else 0.5 if fresh_in else 0.4 if facing >= 30 else 0.1

        def p_switch(klass, *, streak, ko_one, has_absorber):
            if not has_absorber:
                return 0.0
            return 0.1 if klass != "search" else (0.6 if streak and ko_one >= 0.5 else 0.1 if ko_one >= 0.5 else 0.03)

        def p_setup(klass, *, threatened, turn, move_id):
            return 0.05 if threatened else (0.4 if move_id == "tailwind" and (turn or 1) <= 2 else 0.2)

        def p_fakeout(klass, *, turn):
            return 0.8 if (turn or 1) <= 1 else (0.5 if klass == "search" else 0.2)

        return p_protect, p_switch, p_setup, p_fakeout


def class_of(memory: Any) -> str:
    """'search' / 'llm' / 'random' from memory.opponent_class() (the ops classifier), default 'search'."""
    fn = getattr(memory, "opponent_class", None)
    label = ""
    if callable(fn):
        try:
            result = fn()
            label = str(result[0] if isinstance(result, tuple) else result or "").lower()
        except Exception:  # noqa: BLE001
            label = ""
    if "random" in label or "dead" in label:
        return "random"
    if "llm" in label or "generic" in label:
        return "llm"
    return "search"


def _our_hits_into(sheet: Any, foe_key: str) -> tuple[float, float, float]:
    """(best single-hit KO probability, two-slot KO probability, the biggest hit in % of its current HP)."""
    best_by_slot: dict[int, tuple[float, list]] = {}
    facing = 0.0
    for entry in getattr(sheet, "our_options", None) or []:
        for option in entry.get("options") or []:
            for row in option.get("targets") or []:
                if row.get("side") != "theirs" or species_key(row.get("species")) != foe_key:
                    continue
                dmg = row.get("damage_pct_of_current_hp") or [0, 0]
                p = battle.ko_probability({"damage_pct_of_current_hp": dmg})
                facing = max(facing, float(dmg[1]))
                slot = entry.get("slot", 0)
                if slot not in best_by_slot or (dmg[0] + dmg[1]) > sum(best_by_slot[slot][1]):
                    best_by_slot[slot] = (p, list(dmg))
    ko_one = max((p for p, _ in best_by_slot.values()), default=0.0)
    ko_two = 0.0
    if len(best_by_slot) == 2:
        (_, a), (_, b) = best_by_slot.values()
        ko_two = battle.ko_probability({"damage_pct_of_current_hp": [a[0] + b[0], a[1] + b[1]]})
    return ko_one, ko_two, facing


def predict(sheet: Any, memory: Any, *, roster: dict | None = None, turn: Any = None) -> dict:
    """The code's prediction for each opposing slot and the joint top column.

    Returns ``{"class", "per_slot": {position: {"species", "p": {kind: prob}, "top": kind}}, "top_column": {position:
    kind}, "source": "reply_table" | "priors"}``. With the reply mixture's table on the sheet, its top column wins."""
    klass = class_of(memory)
    p_protect, p_switch, p_setup, p_fakeout = _priors()
    theirs = {t.get("position"): t for t in getattr(sheet, "threats", None) or []}
    out: dict[str, Any] = {"class": klass, "per_slot": {}, "top_column": {}, "source": "priors"}
    unseen = int((((roster or {}).get("theirs") or {}).get("unseen_count")) or 0)
    seen_back = len((((roster or {}).get("theirs") or {}).get("seen_in_back")) or [])
    has_absorber = (unseen + seen_back) > 0
    for position, threat in theirs.items():
        key = species_key(threat.get("species"))
        moves = {data.to_id(m) for m in threat.get("known_moves") or []}
        streak = int((getattr(memory, "opp_protect_streak", None) or {}).get(key, 0))
        fresh_in = key in set(getattr(memory, "opp_fresh_active", None) or [])
        ko_one, ko_two, facing = _our_hits_into(sheet, key)
        p: dict[str, float] = {}
        p["protect"] = p_protect(klass, streak=streak, ko_one=ko_one, ko_two=ko_two, facing=facing, fresh_in=fresh_in, turn=turn) \
            if moves & set(battle.PROTECT_LIKE) else 0.0
        p["switch"] = p_switch(klass, streak=streak, ko_one=ko_one, has_absorber=has_absorber)
        setup = [m for m in moves if m in SETUP_MOVES]
        p["setup"] = max((p_setup(klass, threatened=ko_one >= 0.5, turn=turn, move_id=m) for m in setup), default=0.0)
        p["fakeout"] = p_fakeout(klass, turn=turn) if ("fakeout" in moves and fresh_in) else 0.0
        p["attack"] = max(0.0, 1.0 - min(1.0, p["protect"] + p["switch"] + p["setup"] + p["fakeout"]))
        top = max(KINDS, key=lambda k: p[k])
        out["per_slot"][position] = {"species": threat.get("species"), "p": {k: round(v, 2) for k, v in p.items()}, "top": top}
        out["top_column"][position] = top
    predicted = getattr(memory, "_reply_predicted", None)
    if isinstance(predicted, dict) and predicted.get("turn") == turn and predicted.get("actions"):
        column = {}
        for species, kind, _move in predicted["actions"]:
            position = next((pos for pos, t in theirs.items() if species_key(t.get("species")) == species_key(species)), None)
            if position is not None:
                column[position] = "attack" if kind in ("attack", "status", "pass") else kind
        if column:
            out["top_column"] = column
            out["source"] = "reply_table"
    return out


def prediction_matches(model_prediction: Any, top_column: dict) -> bool:
    """The model's {their_slot_1: kind, their_slot_2: kind} against the code's top column (positions 1 and 2)."""
    if not isinstance(model_prediction, dict) or not top_column:
        return False
    for position, kind in top_column.items():
        stated = str(model_prediction.get(f"their_slot_{position}") or "").lower().replace("_", "").replace(" ", "")
        if stated != kind.replace("_", ""):
            return False
    return True
