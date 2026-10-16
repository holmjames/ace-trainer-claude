"""A "reader" practice opponent for the simulator: it predicts what OUR code brain will play each turn and best-responds.

Live opponents of one kind read our predictable play: they Protect, or switch out, exactly the Pokemon we are about to
hit, turn after turn. None of our other sparring bots do that, so nothing aimed at that weakness could be measured.
This one does it with an oracle no live opponent has: the simulator shows it our seat's observation for every decision
we make (sim/harness.py ``oracle_of``: a player with ``wants_opponent_view = True`` is handed the other seat's state
through ``observe_opponent`` before its own decision of the same step). It never sees our chosen action or any RNG:

    python sim/harness.py --games 20 --p1 code --p2 agent.versions.reader --seed 21
    python sim/draft_eval.py match agent.versions.code_only agent.versions.reader --pairs 100 --seed 21 --pin-tiebreak --out f.jsonl

How it decides:

- A shadow copy of our code brain (``PokemonAgent(None)``, DEFAULT tuning; AGENT_TUNE is never read) is fed our seat's
  every observation (draft, Team Preview, every battle decision), so its match memory follows ours, and its answer to
  each battle state is the prediction: the code brain's top row, our deterministic play. Under ``--pin-tiebreak`` its
  draft / Team Preview tie-break seed is ours too (the pinned value), so it mirrors code-only exactly.
- Draft and Team Preview: the plain code brain.
- Battle: the code brain's own turn (its top row) on its own sheet, then for each of its slots:
  1. the predicted hits into the slot (its own sheet's ``threats``: the foe, that move, that slot; a spread move hits
     both slots) KO it ``hard_ko`` of the time or take ``hard_damage`` % of its HP: it dodges, unless its own move KOs
     the foe aiming at it and lands first. A switch-in that takes those hits for at most ``great_switch`` % (immune,
     or nearly) is the first answer; else Protect (if it has one, its last move was not a Protect, and it can act);
     else the bench Pokemon that takes the hits best (at most ``switch_ratio`` of what the slot would take, KO'd at
     most ``switch_in_ko_max`` of the time);
  2. a slot not dodging that attacks a foe we are predicted to Protect hits the other foe instead (its best attack
     avoiding that target, a spread move included), or else uses its best support / setup move;
  3. a slot the code brain would Protect while nothing of ours is predicted into it attacks instead.
  Replacements after a faint, and any turn without a prediction, play the code brain's turn.

Every answer goes through the agent's own validator (examples/llm/pokemon.py); any error plays the code brain's turn
(and any error in the code brain the agent's own last-resort move). No randomness of its own: the code brain's
choices plus the deterministic rules above. ``reader.stats`` counts its reads (predicted hard hits, dodges by kind,
retargets); every battle decision is in its decision log (kind "turn", key "reader").

Measured (Oct 15, sim/harness.py --games 20 --p1 code --p2 agent.versions.reader --seed 21): the code brain won 6 of
20; the reader Protected or switched 155 of the 272 slots our play hit with a damaging move (95 Protect, 60 switch).
"""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Any

from examples.llm import pokemon as pokemon_adapters
from examples.llm.base import InvalidChoice

from agent.agent import PokemonAgent, _payload, _threatened_by_us
from agent.pokemon import battle as battle_rules
from agent.pokemon import data
from agent.pokemon.memory import observation_dict, species_key

# The knobs of the read; fixed (no environment overrides: a practice opponent the A/B series can trust to stay put).
READ: dict[str, float] = {
    "hard_ko": 0.4,            # a slot dodges when the predicted hits into it KO it this often ...
    "hard_damage": 50.0,       # ... or take this % of its current HP on average
    "great_switch": 15.0,      # a switch-in taking at most this % (avg, of its HP) from those hits beats a Protect
    "switch_ratio": 0.5,       # a switch-in (no Protect left) takes at most this share of what the slot would take ...
    "switch_in_ko_max": 0.25,  # ... and is KO'd by those hits at most this often
    "unprotect_damage": 20.0,  # the code brain's Protect stands when the predicted hits into the slot reach this %
}

SPREAD_TARGETS = ("allAdjacentFoes", "allAdjacent")


class _NullLog:
    """The shadow's decision log: nothing written (the reader's own log says what it predicted)."""

    def write(self, kind: str, **fields: Any) -> None:  # noqa: D401
        return None


def _avg(est: Any) -> float:
    return (float(est[0]) + float(est[1])) / 2.0


def _unwrap(decision: Any) -> Any:
    return getattr(decision, "action", decision)


def predicted_play(template: dict, payload: Any) -> dict[int, dict]:
    """Our predicted turn (our template and our answer: the built payload or {"slot_k": {option, target}}) in the
    reader's frame: per foe board position (1, 2 = our slot 0, 1) {"kind": "move" | "protect" | "switch" | "pass",
    "move", "into": the reader's slots it hits, "target"}."""
    out: dict[int, dict] = {}
    if not isinstance(payload, dict):
        return out
    for i, slot in enumerate(template.get("slots") or []):
        k = slot.get("slot", i)
        answer = payload.get(f"slot_{k}") or {}
        if answer.get("type"):  # the built payload ({"type": "move", "move_id", "target"} / switch / pass)
            opt = answer
        else:  # an answer by option index ({"option", "target"})
            options = slot.get("options") or []
            index = answer.get("option")
            if not isinstance(index, int) or not 0 <= index < len(options):
                continue
            opt = options[index]
        pos = int(k) + 1
        if opt.get("type") == "switch":
            out[pos] = {"kind": "switch", "species": opt.get("species"), "into": []}
            continue
        if opt.get("type") != "move":
            out[pos] = {"kind": "pass", "into": []}
            continue
        move = data.to_id(opt.get("move_id"))
        if move in battle_rules.PROTECT_OPTIONS:
            out[pos] = {"kind": "protect", "move": move, "into": []}
            continue
        target = answer.get("target", 0)
        info = data.move_info(move) or {}
        if info.get("target") in SPREAD_TARGETS:
            into = [0, 1]
        elif target in (1, 2):
            into = [int(target) - 1]
        else:
            into = []
        out[pos] = {"kind": "move", "move": move, "into": into, "target": target}
    return out


class ReaderAgent(PokemonAgent):
    """The code brain that reads our code brain's turn through the simulator's oracle hook (module docstring)."""

    wants_opponent_view = True  # sim/harness.py: show me the other seat's state before each of my decisions

    def __init__(self, read: dict | None = None) -> None:
        super().__init__(None, version="reader")  # DEFAULT tuning, never AGENT_TUNE
        self.read = {**READ, **(read or {})}
        self.shadow = self._new_shadow()
        self._prediction: dict | None = None  # {"turn", "state_version", "forced", "template", "payload"} of our latest state
        self.stats: Counter = Counter()
        self.last_read: dict | None = None  # the last battle decision: the code brain's answer and what the read changed

    @staticmethod
    def _new_shadow() -> PokemonAgent:
        shadow = PokemonAgent(None, version="reader-shadow", log=lambda _line: None)
        shadow.decision_log = _NullLog()
        shadow._capture_dir = None  # AGENT_CAPTURE_DIR is for the reader's own states, never ours
        return shadow

    # -- the oracle hook: our seat's every observation ---------------------------------------------------------------

    def observe_opponent(self, state: Any, context: Any) -> None:
        """Feed our seat's state to the shadow brain; on a battle state its answer is the prediction for that step."""
        try:
            first = state.legal_actions[0] if state.legal_actions else None
            decision = self.shadow.choose_action(state, context)
            if first is not None and first.action_id == "doubles_turn":
                obs = observation_dict(state)
                template = first.input.get("action") or {}
                self._prediction = {"turn": obs.get("turn"), "state_version": getattr(state, "state_version", None),
                                    "forced": any(s.get("force_switch") for s in template.get("slots") or []),
                                    "template": template, "payload": _payload(_unwrap(decision))}
                self.stats["predictions"] += 1
        except Exception as exc:  # noqa: BLE001 — no prediction this step: the code brain's turn plays
            self._prediction = None
            self.stats["shadow_errors"] += 1
            self._say(f"[reader] shadow brain ERROR ({type(exc).__name__}: {exc}); no read this step")

    # -- battle turn: the code brain's row, then the read ------------------------------------------------------------

    def _turn(self, action, state, obs, log):
        base = pokemon_adapters.doubles_choice(action, state)
        template = action.input.get("action") or {}
        sheet = battle_rules.build_sheet(template, obs, self.memory, self.params)
        value, row, answer = base.fallback(), None, None
        for candidate in sheet.candidates:  # the code brain's choice, exactly as code-only plays it
            try:
                value = base.build({"slot_0": candidate["slot_0"], "slot_1": candidate["slot_1"]})
            except InvalidChoice:
                continue
            row, answer = candidate, {"slot_0": dict(candidate["slot_0"]), "slot_1": dict(candidate["slot_1"])}
            break
        summary = row["why"] if row else "Played the computed default turn."
        events: list[str] = []
        record: dict = {}
        forced = any(slot.get("force_switch") for slot in template.get("slots") or [])
        play = self._current_play(obs) if not forced else None
        if answer is not None and play:
            try:
                out, events, record = self.respond(template, obs, sheet, answer, play)
                if events:
                    value = base.build(out)
                    summary = "Reader: " + "; ".join(events) + "."
            except Exception as exc:  # noqa: BLE001 — the code brain's turn stands
                events, record = [], {"error": f"{type(exc).__name__}: {exc}"}
                self.stats["read_errors"] += 1
                self._say(f"[reader] read ERROR ({type(exc).__name__}: {exc}); the code brain's turn plays")
                log.write("error", phase="moving", error=f"reader: {type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        self.last_read = {"turn": obs.get("turn"), "forced": forced, "code": answer, "events": list(events)}
        self.memory.record_turn(turn=obs.get("turn"), payload=_payload(value), model=None, forced=forced,
                                threatened=_threatened_by_us(sheet), doubled=list(getattr(sheet, "focus_threats", None) or []))
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=_payload(value), code_row=row and row.get("name"),
                  code_payload=answer, reader=events, read=record, forced=forced)
        return value, summary

    def _current_play(self, obs: dict) -> dict[int, dict] | None:
        """Our predicted play for this battle turn, or None (no prediction for this turn, or ours was a replacement)."""
        p = self._prediction
        if not p or p.get("forced") or p.get("turn") != obs.get("turn"):
            return None
        play = predicted_play(p["template"], p["payload"])
        return play or None

    def respond(self, template: dict, obs: dict, sheet: Any, answer: dict, play: dict[int, dict]) -> tuple[dict, list[str], dict]:
        """The best response to ``play`` (``predicted_play``) over the code brain's ``answer`` ({"slot_0": {option,
        target}, "slot_1": ...}): (the answer to play, what changed, the read for the log)."""
        R = self.read
        slots = {s.get("slot", 0): s for s in template.get("slots") or []}
        out = {k: dict(v) for k, v in answer.items()}
        events: list[str] = []
        fs = sheet.field
        if fs is None:
            return out, events, {}
        ours = {m.position: m for m in fs.actives if m.side == "mine"}
        theirs = {m.position: m for m in fs.actives if m.side == "theirs"}
        if not ours or not theirs:
            return out, events, {}
        incoming = {n: self.incoming(sheet, play, n, ours) for n in ours}
        record = {"play": {str(p): {k: v for k, v in a.items() if k != "hit"} for p, a in play.items()},
                  "incoming": {str(n): {k: inc[k] for k in ("damage", "ko", "moves")} for n, inc in incoming.items()}}
        changed: set[int] = set()

        def option(n: int) -> dict:
            index = out.get(f"slot_{n}", {}).get("option")
            opts = (slots.get(n) or {}).get("options") or []
            return opts[index] if isinstance(index, int) and 0 <= index < len(opts) else {}

        # 1. dodge the hard hits: a great switch-in, else Protect, else a good switch-in
        hard = [n for n in sorted(ours) if incoming[n]["ko"] >= R["hard_ko"] or incoming[n]["damage"] >= R["hard_damage"]]
        hard.sort(key=lambda n: -(incoming[n]["ko"] * 100 + incoming[n]["damage"]))
        for n in hard:
            self.stats["hard_reads"] += 1
            if self._ko_first(n, sheet, option(n), out[f"slot_{n}"], incoming[n], ours):
                self.stats["stays_for_ko"] += 1
                events.append(f"slot {n} stays: its move KOs the foe aiming at it first")
                continue
            partner = option(1 - n)
            busy = species_key(partner.get("species")) if partner.get("type") == "switch" else None
            picks = [p for p in self.switch_ins(n, slots.get(n) or {}, obs, sheet, ours, theirs, incoming[n]) if species_key(p[1]) != busy]
            protect = self._protect_option(n, slots.get(n) or {}, sheet, ours)
            why = f"{', '.join(incoming[n]['moves'])} into it, {incoming[n]['damage']:.0f}% expected, KO {100 * incoming[n]['ko']:.0f}%"
            great = next((p for p in picks if p[2] <= R["great_switch"] and p[3] <= 0.05), None)
            good = next((p for p in picks if p[2] <= R["switch_ratio"] * min(incoming[n]["damage"], 100.0) and p[3] <= R["switch_in_ko_max"]), None)
            if great is not None:
                out[f"slot_{n}"] = {"option": great[0], "target": 0}
                events.append(f"slot {n} switches to {great[1]} (reads {why}; it takes {great[2]:.0f}%)")
                self.stats["dodge_switch"] += 1
            elif protect is not None:
                out[f"slot_{n}"] = {"option": protect, "target": 0}
                events.append(f"slot {n} Protects (reads {why})")
                self.stats["dodge_protect"] += 1
            elif good is not None:
                out[f"slot_{n}"] = {"option": good[0], "target": 0}
                events.append(f"slot {n} switches to {good[1]} (reads {why}; it takes {good[2]:.0f}%)")
                self.stats["dodge_switch"] += 1
            else:
                self.stats["no_dodge"] += 1
                continue
            changed.add(n)
        # 2-3. the slots still attacking: around a predicted Protect; a needless Protect of the code brain's attacks
        protecting = {pos for pos, a in play.items() if a.get("kind") == "protect" and pos in theirs}
        for n in sorted(ours):
            if n in changed:
                continue
            opt = option(n)
            if opt.get("type") != "move":
                continue
            move = data.to_id(opt.get("move_id"))
            if move in battle_rules.PROTECT_OPTIONS:
                if incoming[n]["damage"] < R["unprotect_damage"] and incoming[n]["ko"] <= 0.0:
                    attack = self.attack_answer(n, slots.get(n) or {}, sheet, avoid=protecting)
                    if attack:
                        out[f"slot_{n}"] = attack[0]
                        changed.add(n)
                        events.append(f"slot {n} attacks with {attack[1]} instead of Protecting (nothing of theirs aimed at it)")
                        self.stats["unprotect"] += 1
                continue
            target = out[f"slot_{n}"].get("target")
            info = data.move_info(move) or {}
            if not protecting or target not in protecting or info.get("target") in SPREAD_TARGETS:
                continue
            attack = self.attack_answer(n, slots.get(n) or {}, sheet, avoid=protecting)
            if attack:
                out[f"slot_{n}"] = attack[0]
                events.append(f"slot {n} hits the other foe with {attack[1]} (reads a Protect at {target})")
                self.stats["retarget"] += 1
            else:
                support = self.support_answer(n, slots.get(n) or {}, sheet, ours, theirs, avoid=protecting)
                if not support:
                    continue
                out[f"slot_{n}"] = support[0]
                events.append(f"slot {n} uses {support[1]} (reads a Protect at {target})")
                self.stats["support"] += 1
            changed.add(n)
        record["changed"] = sorted(changed)
        return out, events, record

    # -- the read ----------------------------------------------------------------------------------------------------

    def incoming(self, sheet: Any, play: dict[int, dict], n: int, ours: dict[int, Any]) -> dict:
        """The predicted hits into our slot ``n``, priced on our own sheet: expected damage (% of current HP), KO chance,
        the hits [(foe position, move, hit row)]."""
        hits = []
        for pos, act in sorted(play.items()):
            if act.get("kind") != "move" or n not in act.get("into", []):
                continue
            threat = next((t for t in sheet.threats if t.get("position") == pos), None)
            if threat is None:
                continue
            hit = next((h for h in threat.get("hits") or [] if h.get("move") == act["move"] and h.get("into_slot") == n), None)
            if hit is None or hit["damage_pct_of_current_hp"][1] <= 0:
                continue
            hits.append((pos, act["move"], hit, sheet.foe_acts(pos)))
        damage, survive = 0.0, 1.0
        for _, _, hit, acts in hits:
            damage += acts * min(_avg(hit["damage_pct_of_current_hp"]), 100.0) * float(hit.get("hit_chance", 1.0))
            survive *= 1.0 - acts * battle_rules.ko_probability(hit, with_accuracy=True)
        ko = 1.0 - survive
        if len(hits) >= 2:  # both foes into the slot: their hits add up
            (_, _, h1, a1), (_, _, h2, a2) = hits[:2]
            lo = h1["damage_pct_of_current_hp"][0] + h2["damage_pct_of_current_hp"][0]
            hi = h1["damage_pct_of_current_hp"][1] + h2["damage_pct_of_current_hp"][1]
            ko = max(ko, a1 * a2 * float(h1.get("hit_chance", 1.0)) * float(h2.get("hit_chance", 1.0)) * battle_rules.linear_ko(lo, hi))
        return {"damage": round(min(damage, 200.0), 1), "ko": round(min(1.0, ko), 3), "moves": [h[1] for h in hits],
                "hits": [(pos, move) for pos, move, _, _ in hits]}

    def _ko_first(self, n: int, sheet: Any, opt: dict, chosen: dict, inc: dict, ours: dict[int, Any]) -> bool:
        """Our slot's move KOs (every roll) a foe aiming at it, and lands before that foe's hit: it stays and attacks."""
        if opt.get("type") != "move" or not inc["hits"]:
            return False
        entry = self._entry(sheet, n)
        row = next((o for o in entry.get("options") or [] if o.get("option") == chosen.get("option")), None)
        if row is None:
            return False
        target = chosen.get("target")
        aimers = {pos for pos, _ in inc["hits"]}
        species_at = {m.position: species_key(m.species) for m in sheet.field.actives if m.side == "theirs"}
        killed: set = set()
        for r in row.get("targets") or []:
            if r.get("side") != "theirs" or r["damage_pct_of_current_hp"][0] < 100:
                continue
            if r.get("target") == target and target in (1, 2):
                killed.add(target)
            elif r.get("target") == 0:  # a spread move's per-foe row: the foe by species
                killed |= {p for p, k in species_at.items() if k == species_key(r.get("species"))}
        if not aimers <= killed:
            return False
        our_priority = int(row.get("priority") or 0)
        for pos, move in inc["hits"]:
            threat = next((t for t in sheet.threats if t.get("position") == pos), None)
            hit = next((h for h in (threat or {}).get("hits") or [] if h.get("move") == move and h.get("into_slot") == n), None)
            their_priority = int((hit or {}).get("priority") or 0)
            if our_priority != their_priority:
                if our_priority < their_priority:
                    return False
                continue
            if not self._faster(sheet, n, pos):
                return False
        return True

    @staticmethod
    def _faster(sheet: Any, n: int, pos: int) -> bool:
        speed = {(e.get("side"), e.get("position")): e.get("speed", 0) for e in sheet.speed_order or []}
        mine, theirs = speed.get(("mine", n)), speed.get(("theirs", pos))
        if mine is None or theirs is None:
            return False
        return mine < theirs if sheet.trick_room else mine > theirs

    def _protect_option(self, n: int, slot: dict, sheet: Any, ours: dict[int, Any]) -> int | None:
        """The option index of a working Protect for slot ``n``: it has one, it can act, and it did not just Protect."""
        index = next((i for i, o in enumerate(slot.get("options") or []) if o.get("type") == "move"
                      and data.to_id(o.get("move_id")) in battle_rules.PROTECT_OPTIONS), None)
        if index is None or sheet.wake.get(n, 1.0) < 1.0 or n in getattr(sheet, "choice_protect", set()):
            return None
        row = next((o for o in self._entry(sheet, n).get("options") or [] if o.get("option") == index), None)
        if row is not None and row.get("fails"):
            return None
        if self.memory.protect_streak(ours[n].species, n):
            return None
        return index

    def switch_ins(self, n: int, slot: dict, obs: dict, sheet: Any, ours: dict[int, Any], theirs: dict[int, Any],
                   inc: dict) -> list[tuple[int, str, float, float]]:
        """Our bench options for slot ``n``, best first: (option index, species, expected damage % the predicted hits do
        to it on entry, chance they KO it)."""
        fs = sheet.field
        outgoing = ours.get(n)
        our_live = max(1, len(battle_rules._active_summaries(obs.get("team"))))
        found = []
        for index, opt in enumerate(slot.get("options") or []):
            if opt.get("type") != "switch" or not opt.get("species"):
                continue
            summary = battle_rules._find_summary(obs.get("team"), opt["species"])
            if not summary or summary.get("fainted"):
                continue
            x = battle_rules.build_mon({**summary, "boosts": {}}, self.memory.my_cards.get(species_key(opt["species"])), side="mine", position=None)
            if x is None:
                continue
            x.tailwind = "tailwind" in fs.my_side
            damage, survive = 0.0, 1.0
            for pos, move in inc["hits"]:
                foe = theirs.get(pos)
                if foe is None:
                    continue
                hits = battle_rules.entry_hits(foe, x, fs, our_actives=our_live, their_actives=len(theirs), outgoing=outgoing)
                hit = next((h for h in hits if h["move"] == move), None)
                if hit is None:
                    continue  # immune, or the move does nothing to it
                acts = sheet.foe_acts(pos)
                damage += acts * min(_avg(hit["damage_pct_of_current_hp"]), 100.0) * float(hit.get("hit_chance", 1.0)) * x.hp_fraction
                survive *= 1.0 - acts * float(hit.get("ko_chance", 0.0))
            found.append((index, opt["species"], round(damage, 1), round(1.0 - survive, 3)))
        return sorted(found, key=lambda f: (f[3], f[2], f[0]))

    # -- answers -----------------------------------------------------------------------------------------------------

    @staticmethod
    def _entry(sheet: Any, n: int) -> dict:
        return next((e for e in sheet.our_options if e.get("slot") == n), {})

    def attack_answer(self, n: int, slot: dict, sheet: Any, avoid: set[int] | None = None) -> tuple[dict, str] | None:
        """({option, target}, move) for slot ``n``'s best attack by the code brain's pricing, around the foes in ``avoid``
        (a spread move counts only when it still hits a foe outside it)."""
        entry = self._entry(sheet, n)
        avoid = set(avoid or ())
        if avoid:
            others = [p for p in (1, 2) if p not in avoid]
            best = (None, None, -1.0)
            for pos in avoid:
                cand = battle_rules._best_attack(entry, avoid_target=pos, P=self.params)
                if cand[0] and cand[1] is not None and cand[1].get("target") in others and cand[2] > best[2]:
                    best = cand
            spread = self._best_spread(entry, {species_key(m.species) for m in sheet.field.actives if m.side == "theirs" and m.position in avoid})
            if spread and spread[2] > best[2]:
                best = spread
            opt, row, value = best
        else:
            opt, row, value = battle_rules._best_attack(entry, P=self.params)
        if not opt or row is None or value <= 0:
            return None
        options = slot.get("options") or []
        index = opt.get("option")
        if not isinstance(index, int) or not 0 <= index < len(options):
            return None
        legal = options[index].get("targets") or []
        target = row.get("target", 0)
        if legal and target not in legal:
            target = legal[0]
        return {"option": index, "target": target if legal else 0}, opt.get("move") or ""

    @staticmethod
    def _best_spread(entry: dict, avoid_species: set[str]) -> tuple[dict, dict, float] | None:
        """The best spread move by its expected damage to the foes outside ``avoid_species`` alone."""
        best = None
        for option in entry.get("options") or []:
            if option.get("fails"):
                continue
            rows = [r for r in option.get("targets") or [] if r.get("side") == "theirs" and r.get("target") == 0
                    and species_key(r.get("species")) not in avoid_species]
            value = sum(min(_avg(r["damage_pct_of_current_hp"]), 100.0) * float(r.get("hit_chance", 1.0)) for r in rows)
            if value > 0 and (best is None or value > best[2]):
                best = (option, {"target": 0}, value)
        return best

    def support_answer(self, n: int, slot: dict, sheet: Any, ours: dict[int, Any], theirs: dict[int, Any],
                       avoid: set[int]) -> tuple[dict, str] | None:
        """({option, target}, move) for slot ``n``'s best status move (setup, support, disruption of the other foe)."""
        me = ours.get(n)
        if me is None:
            return None
        best = None
        for index, opt in enumerate(slot.get("options") or []):
            if opt.get("type") != "move":
                continue
            move = data.to_id(opt.get("move_id"))
            if move in battle_rules.PROTECT_OPTIONS or data.is_damaging(data.move_info(move) or {"id": move}):
                continue
            value, target = battle_rules.status_value(move, me, theirs, sheet, n, has_ally=len(ours) > 1)
            legal = opt.get("targets") or []
            if value <= 0 or target in avoid:
                continue
            if legal and target not in legal:
                target = next((t for t in legal if t not in avoid), None)
                if target is None:
                    continue
            if best is None or value > best[0]:
                best = (value, {"option": index, "target": target if legal else 0}, move)
        return (best[1], best[2]) if best else None


def create_agent() -> ReaderAgent:
    return ReaderAgent()
