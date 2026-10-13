"""A switch-heavy "pivot" practice opponent for the simulator: the code brain plus a style layer.

None of our sparring bots switch much: the code brain keeps its Pokemon in and attacks, Protects or switches only on a
certain KO. Live opponents of another kind (bulky support teams: Intimidate, Fake Out, redirection and status users
around one sturdy closer) read the attack coming at a slot and send in the teammate that takes it, sometimes both
slots at once, pivot with U-turn or Parting Shot, Fake Out on the turn a user comes in, Protect a slot with no good
switch, and spend turns on Taunt, sleep, Leech Seed, burns and Snarl. This agent plays that way so fixes can be
measured against it (rates measured on three such live games: sim/style_stats.py).

    python sim/harness.py --games 20 --p1 code --p2 agent.versions.pivot --seed 7
    python sim/style_stats.py play --games 20 --p1 code --p2 agent.versions.pivot --seed 7   # the same, plus style stats
    PIVOT_STYLE='{"switch_p": 1.0}' python sim/harness.py ...                                  # knob overrides (STYLE)

How it decides:

- Draft: our drafter's ranking (draft.rank_picks_v2) with a small tilt (``draft_tilt`` log-odds x a 0..1 style score)
  toward bulky cards with Intimidate, Fake Out, pivot moves, disruption or Regenerator. Team Preview: the code brain's.
- Battle: the code brain's turn (its top candidate row, as code-only plays it), then the style layer below may change
  one or both slots. A slot it leaves alone keeps the code brain's choice, so it attacks as well as the code brain.

  1. It reads our likely attack: each foe's best hit (expected damage capped at the HP left plus a KO premium) picks its
     target slot (a spread move hits both; otherwise the slots share the foe's attention by the square of that value).
     A slot is threatened when the aimed hits KO it (``threat_ko``), take ``threat_damage`` % of its HP on average, or
     include a super-effective hit of ``threat_se_damage`` % aimed at it.
  2. A bench Pokemon takes those same aimed moves on entry (battle.entry_hits: its Intimidate, the field after the
     switch). It is a good switch-in when it saves ``switch_gain`` HP points (KOs priced at ``ko_premium``), takes at
     most ``switch_ratio`` of what the slot would, is KO'd by them at most ``switch_in_ko_max`` of the time and by the
     foes' best line into it (the sheet's switch-in preview) at most ``switch_in_entry_ko_max``. Intimidate and Fake Out
     users get ``cycle_bonus``.
  3. One slot threatened with a good switch-in and the other at ``double_threat_scale`` of the bars with its own (judged
     at that share of the switch-in bars): a double switch with ``double_switch_p``; otherwise the most threatened slot with a good switch-in switches with
     ``switch_p``. A slot whose code-brain move KOs a foe, or is a Fake Out, stays.
  4. A slot facing a KO (``protect_ko``) with no good switch-in Protects with ``protect_p`` (not twice in a row, never
     both slots); the code brain's own Protect stands on such a slot and, on a slot not facing a KO, becomes its best
     attack with ``unprotect_p`` (the style Protects far less than the code brain).
  5. A fresh Fake Out user Fakes Out the most dangerous foe it can flinch with ``fake_out_p``.
  6. A pivot move (U-turn, Volt Switch, Flip Turn, Parting Shot, ...) with a bench to go to is used with ``pivot_p``
     when the slot is threatened, or when it is worth ``pivot_margin`` of the slot's best attack.
  7. Disruption (Taunt, sleep, Leech Seed, Will-O-Wisp, Thunder Wave, Encore, ...) is used with ``disrupt_p`` when its
     value (battle.status_value; Leech Seed and Toxic priced here) times ``disrupt_boost`` beats the slot's choice; a
     debuffing attack (Snarl, Icy Wind, ...) gets ``debuff_bonus`` points.
  8. Faint replacements and pivot switch-ins prefer Intimidate / Fake Out users (``cycle_bonus``) among the safe ones.

Measured (sim/style_stats.py definitions; per slot-turn unless "/game"), three live games of the style vs this bot's
20 simulator games against the code brain (seed 7, default knobs), the code brain's own play in the same games last:
hard switches 0.094 / 0.078 / 0.036, double-switch turns 0.33 / 0.05 / 0 per game, Protect 0.062 / 0.115 / 0.138,
pivot moves 0.016 / 0.023 / 0, Fake Out 1.67 / 0.30 / 0.15 per game, disruption 0.078 / 0.014 / 0.009, Intimidate
1.33 / 0.90 / 0.30 per game, game length 11.7 / 7.0 / 7.0 turns. Fake Out and disruption follow what the draft pool
offers; the code brain won 14 of those 20 games.

Determinism: every roll comes from ``random.Random`` seeded with sha256 of (game id, our seat, turn, what is rolled),
the simulator's wall-clock prefix stripped from the game id, so paired series replay the same rolls. Any error in the
style layer plays the code brain's turn; every answer goes through the agent's own validator (examples/llm/pokemon.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import traceback
from typing import Any

from examples.llm import pokemon as pokemon_adapters
from examples.llm.base import InvalidChoice

from agent.agent import PokemonAgent, _payload, _threatened_by_us
from agent.pokemon import battle as battle_rules
from agent.pokemon import data
from agent.pokemon import draft as draft_rules
from agent.pokemon.memory import species_key

# Every knob of the style, with its default. PIVOT_STYLE (a JSON object) overrides them for sweeps.
STYLE: dict[str, float] = {
    # Tuned once (Oct 12) against three live games of the style, from a 16-game simulator batch: switch_p 0.75 -> 0.9,
    # switch_gain 25 -> 20, switch_ratio 0.5 -> 0.6, protect_p 0.6 -> 0.5, disrupt_p 0.6 -> 0.8, disrupt_boost
    # 1.5 -> 2, pivot_margin 0.7 -> 0.6, draft_tilt 0.1 -> 0.5; unprotect_p and double_threat_scale added.
    "draft_tilt": 0.5,               # log-odds added to a draft card per unit of its style score (0..1); 0 = our drafter's ranking
    "threat_ko": 0.4,                # a slot is threatened when the hits aimed at it KO it this often ...
    "threat_damage": 50.0,           # ... or take this % of its current HP on average ...
    "threat_se_damage": 30.0,        # ... or a super-effective hit this big is aimed at it (aim share >= 0.5)
    "double_threat_scale": 0.6,      # next to a threatened slot, the other goes out too when it clears this share of those bars
                                     # (and its switch-in this share of switch_gain, 1/this of switch_ratio)
    "ko_premium": 40.0,              # HP points a KO costs beyond the HP itself, when a switch-in is priced
    "switch_gain": 20.0,             # a switch-in must save at least this many HP points (% of max HP, KO premium included)
    "switch_ratio": 0.6,             # ... take at most this share of the damage the slot would take
    "switch_in_ko_max": 0.25,        # ... be KO'd by the aimed hits at most this often
    "switch_in_entry_ko_max": 0.6,   # ... and by the foes' best line into it (the sheet's switch-in preview) at most this often
    "cycle_bonus": 15.0,             # points for an Intimidate or Fake Out user as the switch-in (also for replacements)
    "switch_p": 0.9,                 # chance a threatened slot with a good switch-in switches
    "double_switch_p": 0.6,          # chance both slots switch at once (one threatened, the other at the scaled bar, each with its own switch-in)
    "protect_ko": 0.5,               # a slot facing a KO this likely, with no good switch-in, Protects ...
    "protect_p": 0.5,                # ... with this chance (the code brain's own Protect of such a slot always stands)
    "unprotect_p": 0.6,              # chance the code brain's Protect of a slot NOT facing a KO becomes the slot's best attack
    "fake_out_p": 0.9,               # chance a fresh Fake Out user Fakes Out
    "pivot_p": 0.7,                  # chance a pivot move is used when its trigger holds
    "pivot_margin": 0.6,             # a damaging pivot move worth this share of the slot's best attack is "near the top"
    "disrupt_p": 0.8,                # chance a disruption move is used when its boosted value beats the slot's choice
    "disrupt_boost": 2.0,            # x the disruption move's value
    "debuff_bonus": 15.0,            # points added to a debuffing attack (Snarl, Icy Wind, Electroweb, ...)
}

PIVOT_MOVES = frozenset({"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception"})
DISRUPTION_MOVES = frozenset({"taunt", "spore", "sleeppowder", "hypnosis", "yawn", "sing", "leechseed", "willowisp",
                              "thunderwave", "glare", "stunspore", "toxic", "encore", "disable"})
DEBUFF_ATTACKS = frozenset({"snarl", "icywind", "electroweb", "strugglebug", "breakingswipe", "lunge", "trailblaze",
                            "mysticalfire", "spiritbreak", "bulldoze", "rocktomb", "nuzzle"})
EXTRA_STATUS_VALUE = {"leechseed": 30.0, "toxic": 30.0, "disable": 15.0}  # status moves battle.status_value prices at 0
CYCLE_ABILITIES = frozenset({"intimidate"})
STYLE_ABILITIES = {"intimidate": 0.35, "regenerator": 0.2, "prankster": 0.1, "friendguard": 0.1}
AIM_KO_PREMIUM = 40.0  # the read's KO premium: the code brain's KO bonus (25) plus its threat-removed share, roughly
SIM_PREFIX = re.compile(r"^sim-\d+-")  # sim/harness.py game ids carry the wall clock: "sim-<unix time>-<game index>"


def style_params(overrides: dict | None = None) -> dict[str, float]:
    """STYLE with ``overrides`` (unknown names are an error)."""
    out = dict(STYLE)
    for key, value in (overrides or {}).items():
        if key not in STYLE:
            raise KeyError(f"unknown pivot style knob {key!r}; known: {sorted(STYLE)}")
        out[key] = float(value)
    return out


def card_style(card: dict) -> float:
    """0..1: how much a draft card plays the bulky-support pivot game (Intimidate, Fake Out, pivot moves, disruption,
    Regenerator, Protect, bulk)."""
    moves = {data.to_id(m) for m in card.get("moves") or []}
    score = STYLE_ABILITIES.get(data.to_id(card.get("ability")), 0.0)
    score += 0.3 if "fakeout" in moves else 0.0
    score += 0.25 if moves & PIVOT_MOVES else 0.0
    score += 0.2 if moves & (DISRUPTION_MOVES | {"snarl", "ragepowder", "followme"}) else 0.0
    score += 0.05 if "protect" in moves or "detect" in moves else 0.0
    base = data.resolve_base_stats(card) or {}
    bulk = (base.get("hp", 0) + base.get("def", 0) + base.get("spd", 0)) / 3.0
    score += 0.3 if bulk >= 110 else 0.2 if bulk >= 95 else 0.0
    return min(1.0, score)


def stable_key(session_id: Any) -> str:
    """The game id with sim/harness.py's wall-clock prefix stripped, so a seeded run's rolls repeat."""
    return SIM_PREFIX.sub("sim-", str(session_id or ""))


def roll(game: str, seat: Any, turn: Any, tag: str) -> float:
    """A uniform draw in [0, 1) from sha256 of the stable state: the same game, seat, turn and tag give the same draw."""
    digest = hashlib.sha256(f"pivot|{game}|{seat}|{turn}|{tag}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random()


def _avg(est: Any) -> float:
    return (float(est[0]) + float(est[1])) / 2.0


def _hit_value(hit: dict, hp_frac: float = 1.0) -> float:
    """What a hit is worth to the attacker, the way the code brain prices its own attacks (battle._expected): expected
    damage in % of the target's max HP (capped at the HP left) plus a KO premium."""
    return min(_avg(hit["damage_pct_of_current_hp"]), 100.0) * float(hit.get("hit_chance", 1.0)) * hp_frac \
        + AIM_KO_PREMIUM * battle_rules.ko_probability(hit, with_accuracy=True)


def _is_spread(move_id: str) -> bool:
    return (data.move_info(move_id) or {}).get("target") in ("allAdjacentFoes", "allAdjacent")


class PivotAgent(PokemonAgent):
    """The code brain with the pivot style layer on its battle turns and a tilted draft (module docstring)."""

    def __init__(self, style: dict | None = None, params: dict | None = None) -> None:
        super().__init__(None, version="pivot", params=params)
        self.style = style_params(style)
        self._context = None

    # -- the stable seed ------------------------------------------------------------------------------------------

    def choose_action(self, state, context):
        self._context = context
        return super().choose_action(state, context)

    def _game(self) -> tuple[str, Any]:
        ctx = self._context
        return stable_key(getattr(ctx, "session_id", None)), getattr(ctx, "seat_position", None)

    def _roll(self, turn: Any, tag: str) -> float:
        game, seat = self._game()
        return roll(game, seat, turn, tag)

    def _match_seed(self) -> int:
        game, seat = self._game()
        return int.from_bytes(hashlib.sha256(f"pivot-match|{game}|{seat}".encode()).digest()[:8], "big")

    # -- draft: our drafter's ranking, tilted toward the style --------------------------------------------------------

    def _draft(self, state, obs, log):
        try:
            if getattr(self.memory, "tiebreak_seed", None) is None:
                self.memory.tiebreak_seed = self._match_seed()  # draft and Team Preview tie-breaks follow the game, not secrets
            seed = draft_rules.tiebreak_seed(self.memory.my_cards, self.memory)
            ranked = draft_rules.rank_picks_v2(state.legal_actions, self.memory, self.params)
            if not ranked:
                return super()._draft(state, obs, log)
            tilt = float(self.style["draft_tilt"])
            values = [s.score + tilt * card_style(self.memory.pool_cards.get(s.card_id) or {}) for s in ranked]
            picks_made = len(self.memory.my_cards) + len(self.memory.opp_cards)
            chosen = ranked[draft_rules.tiebreak(values, float(self.params.get("draft_tiebreak_eps", 0.0)), seed, f"pick:{picks_made}")]
        except Exception as exc:  # noqa: BLE001 — the drafter's own pick, never a lost pick
            log.write("error", phase="draft", error=f"pivot draft: {type(exc).__name__}: {exc}")
            return super()._draft(state, obs, log)
        self.memory.record_pick(chosen.card_id)
        log.write("draft", state_version=state.state_version, card_id=chosen.card_id, species=chosen.species,
                  score=round(chosen.score, 3), style=round(card_style(self.memory.pool_cards.get(chosen.card_id) or {}), 2), pivot=True)
        return chosen.action, f"Drafted {chosen.species}."

    # -- battle turn: the code brain's row, then the style layer --------------------------------------------------------

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
        if answer is not None:
            try:
                styled, events = self.style_turn(template, obs, sheet, answer)
                if events:
                    value = base.build(styled)
                    summary = "Pivot style: " + "; ".join(events) + "."
            except Exception as exc:  # noqa: BLE001 — the code brain's turn stands
                events = []
                self._say(f"[pivot] style layer ERROR ({type(exc).__name__}: {exc}); the code brain's turn plays")
                log.write("error", phase="moving", error=f"pivot style: {type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-2000:])
        forced = any(slot.get("force_switch") for slot in template.get("slots") or [])
        self.memory.record_turn(turn=obs.get("turn"), payload=_payload(value), model=None, forced=forced,
                                threatened=_threatened_by_us(sheet), doubled=list(getattr(sheet, "focus_threats", None) or []))
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=_payload(value), code_row=row and row.get("name"),
                  style=events, pivot=True)
        return value, summary

    def style_turn(self, template: dict, obs: dict, sheet: Any, answer: dict) -> tuple[dict, list[str]]:
        """The style layer over the code brain's ``answer`` ({"slot_0": {option, target}, "slot_1": ...}): the answer to
        play and what changed (empty: the code brain's turn plays)."""
        S = self.style
        slots = {s.get("slot", 0): s for s in template.get("slots") or []}
        out = {k: dict(v) for k, v in answer.items()}
        events: list[str] = []
        turn = obs.get("turn")
        if any(s.get("force_switch") for s in slots.values()):
            return self._replacements(slots, obs, sheet, out)
        fs = sheet.field
        if fs is None:
            return out, events
        ours = {m.position: m for m in fs.actives if m.side == "mine"}
        theirs = {m.position: m for m in fs.actives if m.side == "theirs"}
        if not ours or not theirs:
            return out, events
        reads = self.read_threats(sheet, ours)
        changed: set[int] = set()

        def option(n: int) -> dict:
            index = out[f"slot_{n}"].get("option")
            opts = (slots.get(n) or {}).get("options") or []
            return opts[index] if isinstance(index, int) and 0 <= index < len(opts) else {}

        def kos_with_code_move(n: int) -> bool:
            """The code brain's move from slot ``n`` KOs a foe, or is a Fake Out: the slot stays in and keeps it."""
            opt = option(n)
            if opt.get("type") != "move":
                return False
            if data.to_id(opt.get("move_id")) == "fakeout":
                return True
            row = next((o for o in self._entry(sheet, n).get("options") or [] if o.get("option") == out[f"slot_{n}"]["option"]), None)
            target = out[f"slot_{n}"].get("target")
            return bool(row) and any(r.get("side") == "theirs" and r["damage_pct_of_current_hp"][0] >= 100
                                     and (r.get("target") in (0, target)) for r in row.get("targets") or [])

        # 1-3. switches: read the aimed hits, find the bench Pokemon that takes them
        pressed = [n for n in sorted(ours) if reads[n]["pressed"] and not kos_with_code_move(n)]
        threatened = [n for n in pressed if reads[n]["threatened"]]
        good = {n: self.switch_ins(n, slots.get(n) or {}, obs, sheet, ours, theirs, reads[n]) for n in pressed}
        best_in = {n: picks[0] for n, picks in good.items() if picks}
        threatened.sort(key=lambda n: -(reads[n]["ko"] * 100 + reads[n]["damage"]))
        loose = {n: self.switch_ins(n, slots.get(n) or {}, obs, sheet, ours, theirs, reads[n], relax=S["double_threat_scale"])
                 for n in pressed} if len(pressed) == 2 and any(n in best_in for n in threatened) else {}
        strict = {n: {g[1] for g in picks} for n, picks in good.items()}
        pair = max(((a, b) for a in loose.get(0, []) for b in loose.get(1, []) if species_key(a[1]) != species_key(b[1])
                    and any(n in threatened and pick[1] in strict.get(n, ()) for n, pick in ((0, a), (1, b)))),
                   key=lambda ab: ab[0][2] + ab[1][2], default=None)
        if pair and self._roll(turn, "double_switch") < S["double_switch_p"]:
            for n in (0, 1):
                out[f"slot_{n}"] = {"option": pair[n][0], "target": 0}
                changed.add(n)
            events.append(f"double switch: slot 0 to {pair[0][1]} (reads {reads[0]['why']}), "
                          f"slot 1 to {pair[1][1]} (reads {reads[1]['why']})")
        else:
            for n in threatened:
                other_opt = option(1 - n)
                busy = species_key(other_opt.get("species")) if other_opt.get("type") == "switch" else None
                pick = next((g for g in good.get(n, []) if species_key(g[1]) != busy), None)
                if pick is None:
                    continue
                if self._roll(turn, f"switch:{n}") < S["switch_p"]:
                    out[f"slot_{n}"] = {"option": pick[0], "target": 0}
                    changed.add(n)
                    events.append(f"slot {n} switches to {pick[1]} (reads {reads[n]['why']})")
                break  # one single switch per turn; the other slot keeps its move
        # 4. Protect a slot facing a KO when no good switch-in exists; the code brain's other Protects mostly attack
        for n in sorted(ours):
            if n in changed:
                continue
            opt = option(n)
            protecting = opt.get("type") == "move" and data.to_id(opt.get("move_id")) in battle_rules.PROTECT_OPTIONS
            facing_ko = reads[n]["ko"] >= S["protect_ko"] and n not in best_in and not sheet.alone
            if protecting:
                keep = 1.0 if reads[n]["ko"] >= S["protect_ko"] else 1.0 - S["unprotect_p"]
                attack = self.attack_answer(n, slots.get(n) or {}, sheet)
                if attack and self._roll(turn, f"protect:{n}") >= keep:
                    out[f"slot_{n}"] = attack[0]
                    changed.add(n)
                    events.append(f"slot {n} attacks with {attack[1]} instead of Protecting (reads {reads[n]['why'] or 'nothing aimed at it'})")
                continue
            if not facing_ko or n not in threatened or opt.get("type") == "switch":
                continue
            partner = option(1 - n)
            if partner.get("type") == "move" and data.to_id(partner.get("move_id")) in battle_rules.PROTECT_OPTIONS:
                continue  # never both slots behind Protect
            protect = self._option_index(slots.get(n), battle_rules.PROTECT_OPTIONS)
            if protect is None or self.memory.protect_streak(ours[n].species, n):
                continue
            if self._roll(turn, f"protect:{n}") < S["protect_p"]:
                out[f"slot_{n}"] = {"option": protect, "target": 0}
                changed.add(n)
                events.append(f"slot {n} Protects (no good switch-in; reads {reads[n]['why']})")
        for n in sorted(ours):
            if n in changed or kos_with_code_move(n) or option(n).get("type") != "move":
                continue
            current = data.to_id(option(n).get("move_id"))
            if current in battle_rules.PROTECT_OPTIONS:
                continue
            # 5. Fake Out on the turn it came in
            fake = self.fake_out_answer(n, sheet, ours, theirs)
            if fake and current != "fakeout" and self._roll(turn, f"fakeout:{n}") < S["fake_out_p"]:
                out[f"slot_{n}"] = fake[0]
                changed.add(n)
                events.append(f"slot {n} Fakes Out {fake[1]} on its first turn")
                continue
            # 6. pivot moves
            pivot = self.pivot_answer(n, slots.get(n) or {}, sheet, theirs, reads.get(n) or {})
            if pivot and current not in PIVOT_MOVES and self._roll(turn, f"pivot:{n}") < S["pivot_p"]:
                out[f"slot_{n}"] = pivot[0]
                changed.add(n)
                events.append(f"slot {n} pivots with {pivot[1]}")
                continue
            # 7. disruption
            disrupt = self.disrupt_answer(n, slots.get(n) or {}, sheet, ours, theirs, obs, out[f"slot_{n}"], option(n))
            if disrupt and current != disrupt[1] and self._roll(turn, f"disrupt:{n}") < S["disrupt_p"]:
                out[f"slot_{n}"] = disrupt[0]
                changed.add(n)
                events.append(f"slot {n} disrupts with {disrupt[1]}")
        return out, events

    # -- the read ---------------------------------------------------------------------------------------------------

    def read_threats(self, sheet: Any, ours: dict[int, Any]) -> dict[int, dict]:
        """Per slot of ours: the hits we expect at it this turn ("aimed": [(foe position, move, share, hit)]), their
        expected damage (% of current HP), KO chance, whether that makes it threatened, and a few words why."""
        S = self.style
        use_wake = bool(self.params.get("foe_sleep", 1.0))
        reads = {n: {"aimed": [], "damage": 0.0, "ko": 0.0, "se": False, "threatened": False, "pressed": False, "why": ""} for n in ours}
        singles: dict[int, float] = {n: 1.0 for n in ours}  # chance no single aimed hit KOs the slot

        def value(hit: dict) -> float:
            return _hit_value(hit, ours[hit["into_slot"]].hp_fraction)

        for threat in sheet.threats:
            pos = threat["position"]
            acts = sheet.foe_acts(pos) if use_wake else 1.0
            if acts <= 0:
                continue
            best: dict[int, dict] = {}
            for hit in threat.get("hits") or []:
                if hit.get("move") in ("fakeout", "firstimpression") or hit.get("into_slot") not in ours:
                    continue
                if hit["damage_pct_of_current_hp"][1] <= 0:
                    continue
                n = hit["into_slot"]
                if n not in best or value(hit) > value(best[n]):
                    best[n] = hit
            if not best:
                continue
            top_slot = max(best, key=lambda n: value(best[n]))
            top = best[top_slot]
            if _is_spread(top["move"]):
                shares = {n: 1.0 for n in ours}
                chosen = {n: next((h for h in threat["hits"] if h["move"] == top["move"] and h.get("into_slot") == n), None) for n in ours}
            else:
                weights = {n: value(h) ** 2 for n, h in best.items()}
                total = sum(weights.values()) or 1.0
                shares = {n: w / total for n, w in weights.items()}
                chosen = dict(best)
            for n, hit in chosen.items():
                if hit is None or shares.get(n, 0.0) <= 0:
                    continue
                share = shares[n] * acts
                reads[n]["aimed"].append((pos, hit["move"], share, hit))
                reads[n]["damage"] += share * min(_avg(hit["damage_pct_of_current_hp"]), 100.0) * float(hit.get("hit_chance", 1.0))
                singles[n] *= 1.0 - share * battle_rules.ko_probability(hit, with_accuracy=True)
                if "super effective" in str(hit.get("effect")) and shares[n] >= 0.5 and _avg(hit["damage_pct_of_current_hp"]) >= S["threat_se_damage"]:
                    reads[n]["se"] = True
        for n, read in reads.items():
            ko = 1.0 - singles[n]
            if len(read["aimed"]) >= 2:  # both foes into the slot: their hits add up
                (_, _, s1, h1), (_, _, s2, h2) = sorted(read["aimed"], key=lambda a: -a[2])[:2]
                lo = h1["damage_pct_of_current_hp"][0] + h2["damage_pct_of_current_hp"][0]
                hi = h1["damage_pct_of_current_hp"][1] + h2["damage_pct_of_current_hp"][1]
                ko = max(ko, s1 * s2 * float(h1.get("hit_chance", 1.0)) * float(h2.get("hit_chance", 1.0)) * battle_rules.linear_ko(lo, hi))
            read["ko"] = round(min(1.0, ko), 3)
            read["damage"] = round(read["damage"], 1)
            read["threatened"] = read["ko"] >= S["threat_ko"] or read["damage"] >= S["threat_damage"] or read["se"]
            scale = S["double_threat_scale"]  # the bar a slot must clear to go out alongside a threatened partner
            read["pressed"] = read["threatened"] or read["ko"] >= scale * S["threat_ko"] or read["damage"] >= scale * S["threat_damage"]
            if read["aimed"]:
                main = max(read["aimed"], key=lambda a: a[2])
                read["why"] = f"{main[1]} at it, {read['damage']:.0f}% expected, KO {100 * read['ko']:.0f}%"
        return reads

    def switch_ins(self, n: int, slot: dict, obs: dict, sheet: Any, ours: dict[int, Any], theirs: dict[int, Any],
                   read: dict, relax: float = 1.0) -> list[tuple[int, str, float]]:
        """The good switch-ins for slot ``n``, best first: (option index, species, score) of each bench Pokemon that
        takes the hits aimed at it well enough (module docstring, step 2). ``relax`` < 1 lowers the gain bar and raises
        the damage-share bar by that factor (the partner of a double switch)."""
        S = self.style
        outgoing = ours.get(n)
        if outgoing is None or not read["aimed"]:
            return []
        fs = sheet.field
        our_live = max(1, len(battle_rules._active_summaries(obs.get("team"))))
        taken_out = min(read["damage"], 100.0) * outgoing.hp_fraction + S["ko_premium"] * read["ko"]
        good: list[tuple[int, str, float]] = []
        for index, opt in enumerate(slot.get("options") or []):
            if opt.get("type") != "switch" or not opt.get("species"):
                continue
            key = species_key(opt["species"])
            summary = battle_rules._find_summary(obs.get("team"), opt["species"])
            if not summary or summary.get("fainted"):
                continue
            x = battle_rules.build_mon({**summary, "boosts": {}}, self.memory.my_cards.get(key), side="mine", position=None)
            if x is None:
                continue
            x.tailwind = "tailwind" in fs.my_side
            damage, survive = 0.0, 1.0
            for pos, move, share, _ in read["aimed"]:
                foe = theirs.get(pos)
                if foe is None:
                    continue
                hits = battle_rules.entry_hits(foe, x, fs, our_actives=our_live, their_actives=len(theirs), outgoing=outgoing)
                hit = next((h for h in hits if h["move"] == move), None)
                if hit is None:
                    continue  # immune, or the move does nothing to it
                damage += share * min(_avg(hit["damage_pct_of_current_hp"]), 100.0) * float(hit.get("hit_chance", 1.0))
                survive *= 1.0 - share * float(hit.get("ko_chance", 0.0))
            ko_in = 1.0 - survive
            preview = battle_rules._preview_for(sheet, opt["species"], n)
            entry_ko = float(preview.get("ko_on_entry", 0.0) or 0.0)
            taken_in = min(damage, 100.0) * x.hp_fraction + S["ko_premium"] * ko_in
            gain = taken_out - taken_in
            if gain < relax * S["switch_gain"] or min(damage, 100.0) * x.hp_fraction > min(1.0, S["switch_ratio"] / relax) * min(read["damage"], 100.0) * outgoing.hp_fraction \
                    or ko_in > S["switch_in_ko_max"] or entry_ko > S["switch_in_entry_ko_max"]:
                continue
            if battle_rules.fresh_sleep_target(sheet, opt["species"], theirs, self.memory, outgoing=outgoing):
                continue  # their sleep user would gain a target it does not have now
            score = gain + (S["cycle_bonus"] if self._cycles(x) else 0.0) - 20.0 * entry_ko
            good.append((index, opt["species"], round(score, 1)))
        return sorted(good, key=lambda g: (-g[2], g[0]))

    @staticmethod
    def _cycles(mon: Any) -> bool:
        return data.to_id(mon.ability) in CYCLE_ABILITIES or any(m.get("id") == "fakeout" for m in mon.moves)

    # -- the other triggers -----------------------------------------------------------------------------------------

    @staticmethod
    def _entry(sheet: Any, n: int) -> dict:
        return next((e for e in sheet.our_options if e.get("slot") == n), {})

    @staticmethod
    def _option_index(slot: dict | None, moves: Any) -> int | None:
        return next((i for i, o in enumerate((slot or {}).get("options") or []) if o.get("type") == "move"
                     and data.to_id(o.get("move_id")) in moves), None)

    def _danger_of(self, sheet: Any, pos: int) -> float:
        """How much the foe at ``pos`` threatens us: the sum of its best hit into each of our slots."""
        threat = next((t for t in sheet.threats if t["position"] == pos), None)
        if threat is None:
            return 0.0
        best: dict[int, float] = {}
        for hit in threat.get("hits") or []:
            if hit.get("move") in ("fakeout", "firstimpression"):
                continue
            best[hit["into_slot"]] = max(best.get(hit["into_slot"], 0.0), _hit_value(hit))
        return sum(best.values())

    def attack_answer(self, n: int, slot: dict, sheet: Any) -> tuple[dict, str] | None:
        """({option, target}, move) for slot ``n``'s best attack by the code brain's pricing (battle._best_attack)."""
        opt, row, value = battle_rules._best_attack(self._entry(sheet, n), P=self.params)
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

    def fake_out_answer(self, n: int, sheet: Any, ours: dict[int, Any], theirs: dict[int, Any]) -> tuple[dict, str] | None:
        """({option, target}, foe) for a working Fake Out from slot ``n`` into the most dangerous foe it can flinch."""
        row = next((o for o in self._entry(sheet, n).get("options") or [] if o.get("move") == "fakeout" and not o.get("fails")), None)
        if row is None:
            return None
        best = None
        for r in row.get("targets") or []:
            pos = r.get("target")
            foe = theirs.get(pos)
            if r.get("side") != "theirs" or foe is None or r["damage_pct_of_current_hp"][1] <= 0:
                continue
            if sheet.foe_acts(pos) <= 0 or not battle_rules._flinchable(foe, theirs, sheet.field, attacker=ours.get(n)):
                continue
            danger = self._danger_of(sheet, pos)
            if best is None or danger > best[0]:
                best = (danger, pos, foe.species)
        if best is None:
            return None
        return {"option": row["option"], "target": best[1]}, best[2]

    def pivot_answer(self, n: int, slot: dict, sheet: Any, theirs: dict[int, Any], read: dict) -> tuple[dict, str] | None:
        """({option, target}, move) for a pivot move from slot ``n`` when its trigger holds (module docstring, step 6)."""
        S = self.style
        if not any(o.get("type") == "switch" for o in slot.get("options") or []):
            return None  # nobody to pivot to
        threatened = bool(read.get("threatened")) and read.get("ko", 0.0) < 0.9  # it must live to move
        entry = self._entry(sheet, n)
        best_any = battle_rules._best_attack(entry, P=self.params)
        for index, opt in enumerate(slot.get("options") or []):
            move = data.to_id(opt.get("move_id")) if opt.get("type") == "move" else ""
            if move not in PIVOT_MOVES:
                continue
            legal = opt.get("targets") or []
            row = next((o for o in entry.get("options") or [] if o.get("option") == index), None)
            if row is not None and not row.get("fails"):
                one = battle_rules._best_attack({"options": [row]}, P=self.params)
                if not one[0] or one[1] is None:
                    continue
                near_top = best_any[2] <= 0 or one[2] >= S["pivot_margin"] * best_any[2]
                if not (threatened or near_top):
                    continue
                target = one[1].get("target", 0)
                if legal and target not in legal:
                    target = legal[0]
                return {"option": index, "target": target if legal else 0}, move
            if row is None and move in ("partingshot", "teleport", "chillyreception") and threatened:
                if legal:
                    pos = max((p for p in legal if p in theirs), key=lambda p: self._danger_of(sheet, p), default=legal[0])
                    return {"option": index, "target": pos}, move
                return {"option": index, "target": 0}, move
        return None

    def _seeded(self, obs: dict, foe: Any) -> bool:
        """Is ``foe`` already under Leech Seed (the protocol log, since its last switch-in)?"""
        name = f": {foe.species}"
        seeded = False
        for line in obs.get("protocol_log") or []:
            if not isinstance(line, str) or name not in line:
                continue
            if line.startswith(("|switch|", "|drag|")):
                seeded = False
            elif line.startswith("|-start|") and "Leech Seed" in line:
                seeded = True
            elif line.startswith("|-end|") and "Leech Seed" in line:
                seeded = False
        return seeded

    def disrupt_answer(self, n: int, slot: dict, sheet: Any, ours: dict[int, Any], theirs: dict[int, Any], obs: dict,
                       current: dict, current_opt: dict) -> tuple[dict, str] | None:
        """({option, target}, move) for a disruption move or debuffing attack from slot ``n`` whose boosted value beats
        the slot's current choice (module docstring, step 7)."""
        S = self.style
        me = ours.get(n)
        if me is None:
            return None
        entry = self._entry(sheet, n)
        current_move = data.to_id(current_opt.get("move_id"))
        current_row = next((o for o in entry.get("options") or [] if o.get("option") == current.get("option")), None)
        if current_row is not None:
            current_value = battle_rules._best_attack({"options": [current_row]}, P=self.params)[2]
            if current_move in DEBUFF_ATTACKS:
                current_value += S["debuff_bonus"]
        else:
            current_value = battle_rules.status_value(current_move, me, theirs, sheet, n, has_ally=len(ours) > 1)[0]
        best: tuple[float, dict, str] | None = None
        for index, opt in enumerate(slot.get("options") or []):
            if opt.get("type") != "move":
                continue
            move = data.to_id(opt.get("move_id"))
            legal = opt.get("targets") or []
            if move in DISRUPTION_MOVES:
                value, target = battle_rules.status_value(move, me, theirs, sheet, n, has_ally=len(ours) > 1)
                if move in EXTRA_STATUS_VALUE:
                    target = 0
                    for pos, foe in theirs.items():
                        if move == "leechseed" and ("grass" in foe.types or self._seeded(obs, foe)):
                            continue
                        if move == "toxic" and (foe.status or {"poison", "steel"} & set(foe.types)):
                            continue
                        if pos in legal or not legal:
                            value, target = EXTRA_STATUS_VALUE[move] * max(0.3, foe.hp_fraction), pos
                            break
                if move == "taunt":
                    target = target if target in theirs else 0
                if value <= 0:
                    continue
                if move in ("taunt", "encore", "disable") and target not in theirs:
                    continue
                boosted = value * S["disrupt_boost"]
            elif move in DEBUFF_ATTACKS:
                row = next((o for o in entry.get("options") or [] if o.get("option") == index), None)
                if row is None or row.get("fails"):
                    continue
                one = battle_rules._best_attack({"options": [row]}, P=self.params)
                if not one[0] or one[2] <= 0:
                    continue
                boosted, target = one[2] + S["debuff_bonus"], (one[1] or {}).get("target", 0)
            else:
                continue
            if legal and target not in legal:
                if move in DEBUFF_ATTACKS or move in EXTRA_STATUS_VALUE:
                    target = legal[0]
                else:
                    continue
            if boosted > current_value and (best is None or boosted > best[0]):
                best = (boosted, {"option": index, "target": target if legal else 0}, move)
        return (best[1], best[2]) if best else None

    # -- replacements -----------------------------------------------------------------------------------------------

    def _replacements(self, slots: dict, obs: dict, sheet: Any, out: dict) -> tuple[dict, list[str]]:
        """A faint replacement or a pivot move's switch-in: among the switch-ins the code brain's preview scores within
        ``cycle_bonus`` of its own pick (and not KO'd on entry half the time), an Intimidate or Fake Out user."""
        S = self.style
        events: list[str] = []

        def chosen(n: int) -> dict:
            options = (slots.get(n) or {}).get("options") or []
            index = out.get(f"slot_{n}", {}).get("option")
            return options[index] if isinstance(index, int) and 0 <= index < len(options) else {}

        taken = {species_key(chosen(n).get("species")) for n in slots if chosen(n).get("type") == "switch"}
        for n, slot in sorted(slots.items()):
            if not slot.get("force_switch"):
                continue
            options = slot.get("options") or []
            if chosen(n).get("type") != "switch":
                continue
            index = out[f"slot_{n}"]["option"]
            current = options[index]["species"]
            current_score = float(battle_rules._preview_for(sheet, current, n).get("score", 0.0) or 0.0)
            best = None
            for i, opt in enumerate(options):
                if opt.get("type") != "switch" or i == index or species_key(opt.get("species")) in taken:
                    continue
                card = self.memory.my_cards.get(species_key(opt["species"])) or {}
                if not (data.to_id(card.get("ability")) in CYCLE_ABILITIES or "fakeout" in {data.to_id(m) for m in card.get("moves") or []}):
                    continue
                preview = battle_rules._preview_for(sheet, opt["species"], n)
                if float(preview.get("ko_on_entry", 0.0) or 0.0) >= 0.5:
                    continue
                score = float(preview.get("score", 0.0) or 0.0) + S["cycle_bonus"]
                if score >= current_score and (best is None or score > best[0]):
                    best = (score, i, opt["species"])
            if best is not None:
                taken.discard(species_key(current))
                taken.add(species_key(best[2]))
                out[f"slot_{n}"] = {"option": best[1], "target": 0}
                events.append(f"slot {n} brings in {best[2]} instead of {current} (Intimidate / Fake Out cycling)")
        return out, events


def create_agent() -> PivotAgent:
    raw = os.environ.get("PIVOT_STYLE", "").strip()
    return PivotAgent(style=json.loads(raw) if raw else None)
