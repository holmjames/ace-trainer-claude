"""Per-match memory for the Pokémon agent.

The runtime creates one agent object per match and calls ``choose_action`` on
it turn after turn, so anything stored on this object lives exactly as long as
the match. ``MatchMemory`` is that notebook.

The most valuable thing it remembers: every drafted card is a *complete set*
(species, item, ability, nature, EVs, four moves) and both players' picks are
public. So by the end of the draft we know the opponent's exact six builds —
long before the battle "reveals" anything. ``opp_cards`` is where that lives.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from altruagent import GameState

from . import data

# Values the live server uses for "not known" (Team Preview rosters and unrevealed opponents carry moves: [],
# item: "unknown_item", ability: null, current_hp: 0). They must never overwrite what a drafted card tells us:
# on Oct 7 they did, and every live battle was played blind to the opponent's unused moves and items.
UNKNOWN_VALUES = (None, "", [], {}, "unknown_item", "unknownitem", "unknown")
# Roster-entry fields that describe the card itself (the rest of an entry is battle state: HP, boosts, flags).
CARD_FIELDS = ("species", "name", "types", "base_stats", "item", "ability", "nature", "evs", "ivs", "moves", "level")


def observation_dict(state: GameState) -> dict:
    """The game's observation as a dict (the SDK stringifies it on the model)."""
    raw_obs = state.raw.get("observation") if isinstance(state.raw, dict) else None
    if isinstance(raw_obs, dict):
        return raw_obs
    text = state.observation if isinstance(state.observation, str) else ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return {}


def species_key(name: Any) -> str:
    """Normalize a species name the way the server compares them:
    ``"Urshifu-Rapid-Strike"`` -> ``"urshifurapidstrike"``."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _complete(card: dict) -> bool:
    """A card we know as a full set (drafted or from the catalog), not a roster placeholder or a battle summary."""
    return bool(card.get("card_id")) and len(card.get("moves") or []) >= 4


def _hp_fraction(text: str) -> float | None:
    """``"177/202"`` -> 0.876, ``"45/100"`` -> 0.45, ``"0 fnt"`` -> 0.0; None for anything else."""
    text = str(text or "").strip()
    if text.startswith("0 fnt") or text == "0":
        return 0.0
    m = re.match(r"(\d+)/(\d+)", text)
    if m and int(m.group(2)) > 0:
        return round(int(m.group(1)) / int(m.group(2)), 3)
    return None


@dataclass
class MatchMemory:
    my_seat_key: str | None = None  # the roster/seat key the draft uses for us
    first_drafter: str | None = None
    pool_cards: dict[str, dict] = field(default_factory=dict)  # card_id -> card, every card ever offered
    my_cards: dict[str, dict] = field(default_factory=dict)  # species_key -> card
    opp_cards: dict[str, dict] = field(default_factory=dict)  # species_key -> card
    my_lineup: list[str] = field(default_factory=list)
    opp_lineup_seen: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    fallbacks: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    # -- draft -----------------------------------------------------------------------

    draft_picks: list[dict] = field(default_factory=list)  # every pick seen, in order: species, public_reason, who, seconds (theirs, estimated)
    _draft_turn_started: float | None = None  # server time our current/last draft turn began (deadline - timeout)
    _draft_observed_at: float | None = None  # monotonic time we read that turn's observation
    _draft_think_seconds: float = 0.0  # how long we took on our last pick (local)

    def observe_draft(self, obs: dict, *, my_turn: bool, agent_id: str | None = None) -> None:
        for card in obs.get("available_cards") or []:
            if isinstance(card, dict) and card.get("card_id"):
                self.pool_cards[card["card_id"]] = card
        if self.first_drafter is None and obs.get("first_drafter") is not None:
            self.first_drafter = str(obs.get("first_drafter"))
        rosters = obs.get("rosters")
        if self.my_seat_key is None and agent_id and isinstance(rosters, dict) and agent_id in rosters:
            self.my_seat_key = agent_id  # rosters and current_seat are keyed by agent id
        if my_turn and obs.get("current_seat") is not None:
            self.my_seat_key = str(obs.get("current_seat"))
        self._sync_rosters(obs)
        self._observe_picks(obs, my_turn=my_turn)

    def _observe_picks(self, obs: dict, *, my_turn: bool) -> None:
        """Keep every pick's public reason and estimate how long the opponent took per pick: our turn starts when
        their pick lands (the server's ``decision_deadline_at`` minus ``decision_timeout_seconds``), so the gap
        between two of our turn starts, less our own think time, is their time for the picks in between."""
        started = None
        deadline, timeout = obs.get("decision_deadline_at"), obs.get("decision_timeout_seconds")
        if my_turn and isinstance(deadline, str) and isinstance(timeout, (int, float)):
            try:
                from datetime import datetime

                started = datetime.fromisoformat(deadline.replace("Z", "+00:00")).timestamp() - float(timeout)
            except ValueError:
                started = None
        known = {p.get("pick_number") for p in self.draft_picks}
        new = []
        for pick in obs.get("picks") or []:
            if not isinstance(pick, dict) or pick.get("pick_number") in known:
                continue
            seat = str(pick.get("seat_id")) if pick.get("seat_id") is not None else None
            entry = {"pick_number": pick.get("pick_number"), "species": pick.get("species"), "card_id": pick.get("card_id"),
                     "public_reason": pick.get("public_reason"), "mine": seat is not None and seat == self.my_seat_key,
                     "auto": bool(pick.get("auto")), "seconds": None}
            self.draft_picks.append(entry)
            new.append(entry)
        theirs_new = [p for p in new if not p["mine"]]
        if started is not None and self._draft_turn_started is not None and theirs_new:
            gap = started - self._draft_turn_started - self._draft_think_seconds
            if gap >= 0:
                for p in theirs_new:
                    p["seconds"] = round(gap / len(theirs_new), 2)
        if started is not None:
            self._draft_turn_started = started
            self._draft_observed_at = time.monotonic()
            self._draft_think_seconds = 0.0

    def draft_profile(self) -> dict:
        """How the opponent drafts, for the model and the post-mortem: templated reasons (the same skeleton twice or
        more, species names masked), free text, or none; mean seconds per pick when timing was measurable."""
        theirs = [p for p in self.draft_picks if not p["mine"]]
        names = {str(c.get("species")) for c in self.pool_cards.values() if c.get("species")}
        names |= {str(p.get("species")) for p in self.draft_picks if p.get("species")}
        pattern = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)), re.IGNORECASE) if names else None

        def skeleton(text: str) -> str:
            masked = pattern.sub("<species>", text) if pattern else text
            return re.sub(r"\d+", "#", re.sub(r"\s+", " ", masked)).strip().lower()

        skeletons: dict[str, int] = {}
        for p in theirs:
            reason = str(p.get("public_reason") or "").strip()
            if reason:
                skeletons[skeleton(reason)] = skeletons.get(skeleton(reason), 0) + 1
        counts = {"templated": 0, "free_text": 0, "empty": 0}
        for p in theirs:
            reason = str(p.get("public_reason") or "").strip()
            if not reason:
                kind = "empty"
            elif skeletons.get(skeleton(reason), 0) >= 2 or (len(reason.split()) <= 6 and "<species>" in skeleton(reason)):
                kind = "templated"
            else:
                kind = "free_text"
            counts[kind] += 1
            p["reason_kind"] = kind
        timed = [p["seconds"] for p in theirs if isinstance(p.get("seconds"), (int, float))]
        return {"opponent_picks": len(theirs), "reasons": counts,
                "mean_seconds_per_pick": round(sum(timed) / len(timed), 2) if timed else None,
                "timed_picks": len(timed), "examples": [str(p.get("public_reason") or "")[:120] for p in theirs[:3]]}

    def _sync_rosters(self, obs: dict) -> None:
        rosters = obs.get("rosters")
        if not isinstance(rosters, dict):
            return
        for key, entries in rosters.items():
            if not isinstance(entries, list):
                continue
            mine = self.my_seat_key is not None and str(key) == self.my_seat_key
            if self.my_seat_key is None:
                continue  # can't tell sides yet; wait for our first turn
            target = self.my_cards if mine else self.opp_cards
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                # A card we were offered this match, else the live catalog: the opponent's FIRST pick, when they draft
                # first, is never in our offered list (Oct 7: Dragonite and Incineroar played whole matches as blanks).
                card = self.pool_cards.get(entry.get("card_id") or "") or data.catalog_card(entry.get("card_id"), entry.get("species")) or {}
                merged = {**entry, **card}
                key_species = species_key(merged.get("species"))
                if key_species:
                    target[key_species] = {**target.get(key_species, {}), **merged}

    # -- team preview ----------------------------------------------------------------

    def record_pick(self, card_id: str | None) -> None:
        """Remember the card WE just drafted. The server only talks to us on our own turns, so our final pick never
        shows up in a later draft observation; without this, one of our six reached Team Preview as a blank."""
        card = self.pool_cards.get(str(card_id or ""))
        if card:
            key = species_key(card.get("species"))
            if key:
                self.my_cards[key] = {**self.my_cards.get(key, {}), **card}
        if self._draft_observed_at is not None:
            self._draft_think_seconds = max(0.0, time.monotonic() - self._draft_observed_at)

    def observe_team_preview(self, obs: dict) -> None:
        for side, roster_key in ((self.my_cards, "your_roster"), (self.opp_cards, "opponent_roster")):
            for entry in obs.get(roster_key) or []:
                if not isinstance(entry, dict):
                    continue
                key = species_key(entry.get("species") or entry.get("name"))
                if key and not (side.get(key) or {}).get("moves"):
                    # A roster member we never saw as a full card (our own last pick, the opponent's first pick, or a
                    # restart): recover it from this match's pool, else from the live catalog.
                    card = next((c for c in self.pool_cards.values() if species_key(c.get("species")) == key), None) \
                        or data.catalog_card(species=entry.get("species") or entry.get("name"))
                    if card:
                        side[key] = {**side.get(key, {}), **card}
                self._enrich(side, entry)

    def _enrich(self, side: dict[str, dict], entry: Any) -> None:
        """Fill GAPS in a card from a roster entry (types, base stats, and anything the card lacks). Never overwrite a
        known value, and never take the server's 'unknown' placeholders or battle state (HP, flags) as card facts."""
        if not isinstance(entry, dict):
            return
        key = species_key(entry.get("species") or entry.get("name"))
        if not key:
            return
        known = side.get(key, {})
        fill = {k: v for k, v in entry.items() if k in CARD_FIELDS and v not in UNKNOWN_VALUES and known.get(k) in UNKNOWN_VALUES}
        revealed = entry.get("moves")
        if not _complete(known) and isinstance(revealed, list) and revealed and "moves" not in fill:
            # A set we do not know (never offered, not in the catalog): keep every move the battle has shown so far.
            shown = [m.get("id") if isinstance(m, dict) else m for m in revealed]
            merged = list(known.get("moves") or []) + [m for m in shown if m and m not in (known.get("moves") or [])]
            if merged != list(known.get("moves") or []):
                fill["moves"] = merged
        side[key] = {**known, **fill}

    # -- battle ----------------------------------------------------------------------

    last_active: list[str] = field(default_factory=list)
    fresh_active: list[str] = field(default_factory=list)  # our actives that just came in this turn
    opp_last_active: list[str] = field(default_factory=list)
    opp_fresh_active: list[str] = field(default_factory=list)  # their actives that just came in (Fake Out is live)

    def observe_battle(self, obs: dict) -> None:
        self.observe_opponent_hp(obs)  # uses last turn's opp_last_active, so it runs before the update below
        self.observe_protocol(obs)
        self._rebuild_cards(obs)
        turn = obs.get("turn")
        # Positional: an empty slot stays an empty entry, so slot 1's Pokémon is actives[1] even when slot 0 is empty
        # (compacting the list zeroed every streak and last-turn flag of our last Pokémon standing in slot 1).
        actives = [species_key(entry.get("species") if isinstance(entry, dict) else entry) or None
                   for entry in obs.get("active_pokemon") or []]
        if any(actives):
            self.fresh_active = self._fresh("mine", [a for a in actives if a], self.last_active, turn)
            self.last_active = actives
        opp_actives = [species_key(entry.get("species") if isinstance(entry, dict) else entry) or None
                       for entry in obs.get("opponent_active_pokemon") or []]
        if any(opp_actives):
            self.opp_fresh_active = self._fresh("theirs", [a for a in opp_actives if a], self.opp_last_active, turn)
            self.opp_last_active = opp_actives
        opponent_team = obs.get("opponent_team")
        if isinstance(opponent_team, dict):
            for summary in opponent_team.values():
                if isinstance(summary, dict):
                    key = species_key(summary.get("species"))
                    if key and key not in self.opp_lineup_seen:
                        self.opp_lineup_seen.append(key)
        self._rebuild_lineup(obs)

    def _fresh(self, side: str, actives: list[str], last: list[str], turn: Any) -> list[str]:
        """Which of ``actives`` are on their first turn (Fake Out works): per the protocol log, those that have not
        attempted a move since they last entered the field, exactly as Showdown counts it. That survives a restart
        and a mid-turn replacement decision, which used to "consume" a newcomer's freshness: Incineroar switched in
        during turn 2, the faint replacement decision saw it, and turn 3's sheet no longer feared its Fake Out.
        Without a log: whoever was not active at our last decision (turn 1: everyone), as before."""
        if self.acted_since_entry:
            known = {a: self.acted_since_entry.get(f"{side}:{a}") for a in actives}
            if all(v is not None for v in known.values()):
                return [a for a in actives if not known[a]]
        if last:
            return [a for a in actives if a not in last]
        return list(actives)

    # -- restart safety: rebuild what the draft and Team Preview would have told us ---------------

    def _rebuild_cards(self, obs: dict) -> None:
        """Fill every card visible in a battle observation from the live catalog (``data/cards.json``).

        The runtime starts a fresh worker per match, and a restarted worker starts with an empty notebook: it would
        play the rest of the match blind to every unused move, item and EV spread. The platform's cards are fixed
        sets, so a species seen live before is known for good. Cards learned here carry ``source: "catalog"`` (a
        drafted card is public knowledge; a catalog card is our own record). Nothing drafted is overwritten, and the
        catalog is consulted once per species per match."""
        seen: list[tuple[dict[str, dict], str, dict | None]] = []
        for key_name, side in (("team", self.my_cards), ("opponent_team", self.opp_cards)):
            entries = obs.get(key_name)
            if isinstance(entries, dict):
                seen.extend((side, e["species"], e) for e in entries.values() if isinstance(e, dict) and e.get("species"))
        for key_name, side in (("active_pokemon", self.my_cards), ("opponent_active_pokemon", self.opp_cards)):
            seen.extend((side, e["species"], e) for e in obs.get(key_name) or [] if isinstance(e, dict) and e.get("species"))
        for tag, species_list in self.roster_by_side.items():  # Team Preview's |poke| lines: all six per side
            side = self.my_cards if tag == "mine" else self.opp_cards
            seen.extend((side, species, None) for species in species_list)
        for side, species, entry in seen:
            key = species_key(species)
            if not key:
                continue
            card = side.get(key) or {}
            checked = ("mine:" if side is self.my_cards else "theirs:") + key
            if not _complete(card) and checked not in self._catalog_checked:
                self._catalog_checked.add(checked)
                found = data.catalog_card(species=species)
                if found is None and isinstance(entry, dict) and entry.get("name"):
                    found = data.catalog_card(species=entry.get("name"))
                if found:
                    side[key] = {**card, **found, "source": "catalog"}
                elif not card.get("source"):
                    # Never silent, never blank: a placeholder from the dex (typing, base stats, 0 EVs) plus the moves
                    # the battle has shown; the sheet keeps computing for it. One loud line per card per match.
                    info = data.species_info(species) or (data.species_info(entry.get("name")) if isinstance(entry, dict) else None) or {}
                    side[key] = {**card, "species": card.get("species") or (info.get("name") if info else species), "source": "placeholder",
                                 "placeholder": True, **({"types": [str(t).upper() for t in info.get("types") or []],
                                                          "base_stats": dict(info.get("base_stats") or {})} if info else {})}
                    self.loud(f"[memory] CATALOG GAP: {species} ({'ours' if side is self.my_cards else 'theirs'}) is not in data/cards.json; "
                              "playing it as a placeholder set (dex typing and base stats, revealed moves only). Run scripts/pool_from_live.py.")
            if entry is not None:
                self._enrich(side, entry)
            card = side.get(key)
            if card and (card.get("types") in UNKNOWN_VALUES or card.get("base_stats") in UNKNOWN_VALUES):
                info = data.species_info(card.get("species") or species)  # a reserve nobody has seen: the dex, spelled as the server does
                if info:
                    self._enrich(side, {"species": species, "types": [str(t).upper() for t in info.get("types") or []],
                                        "base_stats": dict(info.get("base_stats") or {})})

    def _rebuild_lineup(self, obs: dict) -> None:
        """Our four for this battle when Team Preview is not in memory: everyone of ours the protocol log has seen on
        the field, plus the fainted and the switches the server offers now. Only when that makes exactly four."""
        if self.my_lineup or not self.entered_by_turn:
            return
        brought = {sp for entries in self.entered_by_turn.values() for tag, sp in entries if tag == "mine"}
        brought.update(a for a in self.last_active if a)
        team = obs.get("team")
        if isinstance(team, dict):
            brought.update(species_key(s.get("species")) for s in team.values() if isinstance(s, dict) and s.get("fainted") and s.get("species"))
        for options in obs.get("available_switches") or []:
            for o in options or []:
                if isinstance(o, dict) and o.get("species"):
                    brought.add(species_key(o["species"]))
        brought.discard("")
        if len(brought) == 4:
            self.my_lineup = sorted(brought)

    last_payload: dict | None = None
    # Rebuilt from the protocol log every turn (restart-safe): who entered the field on which turn, both sides'
    # rosters from Team Preview, and every move each Pokémon used by turn.
    entered_by_turn: dict[int, list[tuple[str, str]]] = field(default_factory=dict)  # turn -> [("mine"/"theirs", species)]
    roster_by_side: dict[str, list[str]] = field(default_factory=dict)  # "mine"/"theirs" -> species as |poke| spelled them
    moves_by_turn: dict[str, dict[str, dict[int, str]]] = field(default_factory=dict)  # side -> species -> {turn: move}
    acted_since_entry: dict[str, bool] = field(default_factory=dict)  # "mine:<species>"/"theirs:<species>" -> tried a move since it last came in
    protect_failed: dict[str, dict[str, set[int]]] = field(default_factory=dict)  # side -> species -> turns its Protect FAILED (counter reset)
    fallen_by_side: dict[str, int] = field(default_factory=dict)  # "mine"/"theirs" -> faints so far (Supreme Overlord, Last Respects)
    overlord_fallen: dict[str, int] = field(default_factory=dict)  # "mine:<species>"/"theirs:<species>" -> fallenN Supreme Overlord announced on entry
    hazards_by_side: dict[str, dict[str, Any]] = field(default_factory=dict)  # "mine"/"theirs" -> entry hazards on that side
    log_turn: int = 0  # the last |turn| marker seen in the protocol log (0: no log)
    _catalog_checked: set[str] = field(default_factory=set)  # species already looked up in the catalog this match
    opp_items_lost: list[str] = field(default_factory=list)  # species_keys of opponents whose item is gone (protocol log -enditem)
    unburden_active: list[str] = field(default_factory=list)  # "mine:<species>"/"theirs:<species>" whose item was lost on the field (Unburden)
    opp_protect_streak: dict[str, int] = field(default_factory=dict)  # their species -> Protects in a row up to last turn (protocol log)
    paradox_active: dict[str, str] = field(default_factory=dict)  # "mine:<species>"/"theirs:<species>" -> stat a running Protosynthesis/Quark Drive boosts
    opp_last_move: dict[str, str | None] = field(default_factory=dict)  # species_key -> last move it used since it last switched in (live server protocol log)
    last_opp_hp: dict[str, float] = field(default_factory=dict)  # species_key -> hp fraction seen last turn
    opp_protected_last_turn: list[str] = field(default_factory=list)  # inferred: we hit it, its HP did not move

    PROTECT_LIKE = frozenset({"protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "wideguard", "quickguard", "obstruct", "kingsshield"})
    SETUP_MOVES = frozenset({"nastyplot", "calmmind", "swordsdance", "dragondance", "bulkup", "quiverdance", "tailwind", "trickroom",
                             "icywind", "electroweb", "lightscreen", "reflect"})

    # Class signals and the turn digest, all rebuilt from the protocol log (restart-safe).
    players: dict[str, str] = field(default_factory=dict)  # "mine"/"theirs" -> the player name Showdown prints (|player| lines)
    their_slow_turns: dict[int, int] = field(default_factory=dict)  # turn -> fewest seconds the opponent had left when warned
    my_slow_turns: dict[int, int] = field(default_factory=dict)
    my_bank_seen: dict[int, float] = field(default_factory=dict)  # turn -> our bank per "Time left: ... | N sec total"
    hp_by_turn: dict[int, dict[str, float]] = field(default_factory=dict)  # turn -> "mine:<species>"/"theirs:<species>" -> hp fraction after the turn
    targets_by_turn: dict[str, dict[int, list[tuple[str, str]]]] = field(default_factory=dict)  # side -> turn -> [(attacker, target species)]
    faints_by_turn: dict[int, list[str]] = field(default_factory=dict)  # turn -> tags that fainted
    their_voluntary_switches: list[int] = field(default_factory=list)  # turns on which they switched without a faint or a pivot move
    their_protect_attempts: int = 0
    their_repeat_protects: int = 0  # a Protect attempted the turn after a Protect (the search bot never does)
    say: Any = field(default=None, repr=False, compare=False)  # a callable for loud one-line notices (the agent's console)

    def loud(self, line: str) -> None:
        if callable(self.say):
            try:
                self.say(line)
            except Exception:  # noqa: BLE001
                pass

    def record_turn(self, **fields: Any) -> None:
        """Remember what we played (``payload``), who stood where, and what the sheet told us about this turn:
        ``threatened`` (their species we could KO) and ``doubled`` (our slot both foes could KO) feed the tendency tally."""
        fields.setdefault("actives", list(self.last_active))  # who stood in each slot when this was played
        self.turns.append(fields)
        payload = fields.get("payload")
        if isinstance(payload, dict):
            self.last_payload = payload

    def protect_streak(self, species: str, slot: int) -> int:
        """How many turns in a row the Pokémon now in ``slot`` has just used a Protect-like move (0 = none).
        Each repeat succeeds with probability 1/3 of the previous one, so a streak of 2 means the next try works 1 in 9.
        A replacement decision after a faint (the partner 'passes' mid-turn) is not a turn of its own and is skipped:
        counting it reset the streak on Oct 7, and a second Protect in a row was recommended as if it were fresh."""
        me = species_key(species)
        if self.log_turn:
            # The protocol log says what actually ran, as Showdown counts it (a Protect we clicked but never got off,
            # asleep or Taunted, starts no streak), and it survives a restart of this process mid-battle.
            return self.logged_streak("mine", me)
        streak = 0
        for fields in reversed(self.turns):
            if fields.get("forced"):
                continue
            payload = fields.get("payload") or {}
            actives = fields.get("actives") or []
            choice = payload.get(f"slot_{slot}") or {}
            if len(actives) <= slot or actives[slot] != me:
                break
            if choice.get("type") == "move" and species_key(choice.get("move_id")) in self.PROTECT_LIKE:
                streak += 1
            else:
                break
        return streak

    def logged_streak(self, side: str, species: str, *, before_turn: int | None = None) -> int:
        """Consecutive Protect-like moves ``species`` (on ``side``) used per the protocol log, ending with the turn
        before ``before_turn`` (default: the turn that just finished)."""
        key = species_key(species)
        by_turn = (self.moves_by_turn.get(side) or {}).get(key) or {}
        failed = (self.protect_failed.get(side) or {}).get(key) or set()
        n, t = 0, (before_turn if before_turn is not None else self.log_turn) - 1
        # A Protect that failed resets Showdown's counter: the next one is guaranteed again, so the streak is 0.
        while by_turn.get(t) in self.PROTECT_LIKE and t not in failed:
            n, t = n + 1, t - 1
        return n

    def fallen(self, side: str) -> int:
        """How many of ``side``'s ("mine"/"theirs") Pokémon have fainted so far, from the protocol log's |faint| lines
        (Supreme Overlord: +10% per fallen ally when Kingambit enters; Last Respects: +50 power each)."""
        return int(self.fallen_by_side.get(side, 0))

    # -- the opponent's class, its clock, and what happened lately -------------------------------------------------

    def opponent_class(self) -> str:
        """``"search_bot" | "generic_llm" | "random" | "unknown"`` (a plain string: the core lane's reply mixture reads it too;
        ``opponent_evidence()`` has the numbers) from name-free signals: draft reason length (free
        text 177-216 chars vs 28-42-char templates), pick timing (tiebreak), their ``|inactive|`` warnings (the LLM
        was slow on 6 of 7 turns, the search bot on 0 of 16), repeated Protect attempts (LLM 1 of 2, search bot 0 of
        9), voluntary switches (LLM 1 of 14), auto picks and timeouts (dead/random). Default: search_bot, the field's
        strongest class, until the evidence says otherwise."""
        return self.opponent_evidence()["class"]

    def opponent_evidence(self) -> dict:
        """The class and the signals behind it (see ``opponent_class``)."""
        theirs = [p for p in self.draft_picks if not p["mine"]]
        reasons = [str(p.get("public_reason") or "").strip() for p in theirs]
        lengths = [len(r) for r in reasons if r]
        mean_len = sum(lengths) / len(lengths) if lengths else 0.0
        profile = self.draft_profile() if theirs else {"reasons": {"templated": 0, "free_text": 0, "empty": 0}, "mean_seconds_per_pick": None}
        turns_seen = max(self.log_turn - 1, 0)
        slow = len(self.their_slow_turns)
        auto = sum(1 for p in theirs if p.get("auto"))
        # A constant or generic reason ("Move submitted." on every pick, Oct 8 evening: 15 chars; anything under 16)
        # says nothing about the drafter: neither a template (the search bot's run 28-42 chars) nor free text (an LLM's
        # 177-216). Only the other signals count in that bucket.
        distinct = {r.lower() for r in reasons if r}
        constant_reason = bool(lengths) and (len(distinct) == 1 and len(lengths) >= 2 or mean_len < 16)
        evidence = {"draft_reason_mean_chars": round(mean_len, 1), "draft_reasons": profile["reasons"], "mean_seconds_per_pick": profile.get("mean_seconds_per_pick"),
                    "constant_reason": constant_reason, "slow_turns": slow, "turns_seen": turns_seen, "repeat_protects": self.their_repeat_protects,
                    "protect_attempts": self.their_protect_attempts, "voluntary_switches": len(self.their_voluntary_switches), "auto_picks": auto}
        llm = search = random_ = 0
        if constant_reason:
            pass  # the reason text is not evidence in this bucket
        elif mean_len >= 100:
            llm += 2
        elif lengths and mean_len <= 60 and (profile["reasons"]["templated"] >= 2 or len(lengths) <= 1):
            search += 2
        if turns_seen >= 2:
            if slow / turns_seen >= 0.3:
                llm += 2
            elif slow == 0 and turns_seen >= 3:
                search += 1
        if self.their_repeat_protects:
            llm += 1
        elif self.their_protect_attempts >= 2:
            search += 1
        if len(self.their_voluntary_switches) >= 2:
            llm += 1
        secs = profile.get("mean_seconds_per_pick")
        if isinstance(secs, (int, float)):
            llm += 0.5 if secs >= 2.5 else 0
            search += 0.5 if secs < 1.5 else 0
        if auto >= 2 or (theirs and not lengths and auto):
            random_ += 3
        if not any(reasons) and theirs and turns_seen >= 3 and self.their_protect_attempts == 0 and slow == 0:
            random_ += 1
        evidence["scores"] = {"search_bot": search, "generic_llm": llm, "random": random_}
        best = max(("search_bot", search), ("generic_llm", llm), ("random", random_), key=lambda t: t[1])
        if constant_reason and best[1] <= 0:
            label = "unknown"  # nothing but a constant reason so far: no read until the battle shows one
        else:
            label = best[0] if best[1] > 0 else "search_bot"
            if best[0] != "search_bot" and best[1] == search:
                label = "search_bot"  # ties go to the strongest class
        evidence["class"] = label
        return evidence

    def opponent_bank_estimate(self) -> dict:
        """Their bank, estimated from Showdown's warnings about THEIR clock ("<name> has 30/20/10 seconds left": at
        least 25/35/45 s of the turn used); a turn without a warning counts as a fast one (~6 s). The live server
        never shows the opponent's bank directly."""
        used = 0.0
        for t in range(1, max(self.log_turn, 1)):
            left = self.their_slow_turns.get(t)
            used += 6.0 if left is None else (45.0 if left <= 10 else 35.0 if left <= 20 else 25.0 + 3.0)
        bank = max(0.0, 420.0 - used)
        return {"bank_estimate_s": round(bank, 1), "used_estimate_s": round(used, 1), "slow_turns": sorted(self.their_slow_turns),
                "turns": max(self.log_turn - 1, 0), "our_bank_s": self.my_bank_seen.get(self.log_turn)}

    def last_turns(self, n: int = 3, *, limit: int = 600) -> str:
        """A compact digest of the last ``n`` finished turns: what each side did, who came in, who fainted, HP after.
        Species as the log spells them (keys)."""
        if not self.log_turn:
            return ""
        lines = []
        for t in range(max(1, self.log_turn - n), self.log_turn):
            parts = []
            for side, label in (("mine", "we"), ("theirs", "they")):
                acts = []
                for species, by_turn in (self.moves_by_turn.get(side) or {}).items():
                    move = by_turn.get(t)
                    if not move:
                        continue
                    target = next((tgt for who, tgt in (self.targets_by_turn.get(side) or {}).get(t, []) if who == species), None)
                    failed = t in ((self.protect_failed.get(side) or {}).get(species) or set())
                    acts.append(f"{species} {move}" + (f"->{target}" if target else "") + (" [failed]" if failed else ""))
                entered = [sp for tag, sp in self.entered_by_turn.get(t, []) if tag == side]
                if entered:
                    acts.append("in: " + ", ".join(entered))
                if acts:
                    parts.append(f"{label}: " + ", ".join(acts))
            fainted = [tag for tag in self.faints_by_turn.get(t, [])]
            if fainted:
                parts.append("KO: " + ", ".join(fainted))
            hp = self.hp_by_turn.get(t) or {}
            if hp:
                parts.append("HP after: " + ", ".join(f"{tag.split(':', 1)[1]} {int(round(v * 100))}%" for tag, v in sorted(hp.items())))
            lines.append(f"T{t}: " + "; ".join(parts) if parts else f"T{t}: (nothing logged)")
        text = " | ".join(lines)
        return text if len(text) <= limit else text[:limit - 1] + "…"

    def opp_tendencies(self) -> dict:
        """How this opponent has answered what the sheet said about each turn we recorded: Protects when we threatened
        a KO (x of y), switches out of a threatened KO, doubles into one of our slots, setups on a free turn."""
        protect = [0, 0]
        switch = [0, 0]
        doubles = [0, 0]
        setups = [0, 0]
        for fields in self.turns:
            t = fields.get("turn")
            if not isinstance(t, int) or t >= self.log_turn or fields.get("forced"):
                continue
            threatened = [species_key(s) for s in fields.get("threatened") or []]
            their_moves = {sp: by.get(t) for sp, by in (self.moves_by_turn.get("theirs") or {}).items() if by.get(t)}
            entered = {sp for tag, sp in self.entered_by_turn.get(t, []) if tag == "theirs"}
            for sp in threatened:
                protect[1] += 1
                if their_moves.get(sp) in self.PROTECT_LIKE:
                    protect[0] += 1
                switch[1] += 1
                if sp not in their_moves and entered and t in self.their_voluntary_switches:
                    switch[0] += 1
            tgts = [tgt for _, tgt in (self.targets_by_turn.get("theirs") or {}).get(t, [])]
            ours_hit = [tgt for tgt in tgts if any(tgt == (a or "") for a in fields.get("actives") or [])]
            if len(ours_hit) >= 2:
                doubles[1] += 1
                if len(set(ours_hit)) == 1:
                    doubles[0] += 1
            elif len(tgts) >= 2:
                doubles[1] += 1
            free = [sp for sp, mv in their_moves.items() if sp not in threatened]
            if free:
                setups[1] += 1
                if any(their_moves[sp] in self.SETUP_MOVES for sp in free):
                    setups[0] += 1
        return {"protects_when_threatened": f"{protect[0]}/{protect[1]}", "switches_out_of_ko": f"{switch[0]}/{switch[1]}",
                "doubles_into_one_slot": f"{doubles[0]}/{doubles[1]}", "setups_on_a_free_turn": f"{setups[0]}/{setups[1]}",
                "repeat_protects": self.their_repeat_protects, "voluntary_switches": len(self.their_voluntary_switches)}

    def hazards(self, side: str) -> dict[str, Any]:
        """Entry hazards lying on ``side`` ("mine"/"theirs"): ``{"spikes": layers, "stealthrock": bool, "stickyweb":
        bool, "toxicspikes": layers}``, from -sidestart/-sideend lines. Match bcbaabe2: Milotic switched into Spikes
        and a Wood Hammer and was lost because nothing knew the Spikes were there."""
        current = self.hazards_by_side.get(side) or {}
        return {"spikes": int(current.get("spikes", 0)), "stealthrock": bool(current.get("stealthrock", False)),
                "stickyweb": bool(current.get("stickyweb", False)), "toxicspikes": int(current.get("toxicspikes", 0))}

    def our_protect_last_turn(self, species: str, slot: int) -> bool:
        """Did the Pokémon now in ``slot`` use a Protect-like move last turn? (Consecutive Protect fails 2/3 of the time.)"""
        if not self.last_payload or not self.last_active or len(self.last_active) <= slot:
            return False
        if species_key(species) != self.last_active[slot]:
            return False  # a different Pokémon is in the slot now
        choice = self.last_payload.get(f"slot_{slot}") or {}
        return choice.get("type") == "move" and species_key(choice.get("move_id")) in {"protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "wideguard"}

    def observe_protocol(self, obs: dict) -> None:
        """Read the Showdown protocol log the live server includes (``observation["protocol_log"]``). From it:

        - the last move each opposing Pokémon used since it last entered the field: a Choice item locks its holder into
          that move until it switches ("Urshifu is locked into Surging Strikes");
        - which opposing items are gone (-enditem: a popped Air Balloon makes Earthquake hit again; Knock Off; berries);
        - which Protosynthesis / Quark Drive boosts are running, on both sides (-start ... protosynthesisatk). Booster
          Energy is consumed the moment it activates, but the boost lasts until the holder leaves the field, so the
          item alone can't tell us;
        - both rosters (Team Preview's |poke| lines), who entered the field on which turn, and every move used by
          turn on both sides: what a worker started mid-battle needs to rebuild the notebook.

        The log is cumulative, so everything is rebuilt from scratch every turn; without a log the tables stay empty."""
        log = obs.get("protocol_log")
        if not isinstance(log, list) or not log:
            return
        mine = {str(k).split(":")[0] for k in (obs.get("team") or {}) if isinstance(k, str) and ":" in k}
        if len(mine) != 1:
            return
        my_side = mine.pop()  # "p1" or "p2"
        last: dict[str, str | None] = {}
        lost: set[str] = set()
        paradox: dict[str, str] = {}  # "mine:<species>" / "theirs:<species>" -> boosted stat
        unburden: set[str] = set()  # same tags: lost its item during the current stint on the field
        nick_to_species: dict[str, str] = {}
        turn = 0
        moves_by_turn: dict[str, dict[str, dict[int, str]]] = {"mine": {}, "theirs": {}}  # side -> species -> {turn: move}
        entered: dict[int, list[tuple[str, str]]] = {}
        acted: dict[str, bool] = {}
        rosters: dict[str, list[str]] = {"mine": [], "theirs": []}
        failed_protect: dict[str, dict[str, set[int]]] = {"mine": {}, "theirs": {}}
        pending_protect: tuple[str, str, int] | None = None  # (side, species, turn) of the Protect-like move just announced
        fallen: dict[str, int] = {"mine": 0, "theirs": 0}
        overlord: dict[str, int] = {}
        hazards: dict[str, dict[str, Any]] = {"mine": {}, "theirs": {}}
        players: dict[str, str] = {}
        slow: dict[str, dict[int, int]] = {"mine": {}, "theirs": {}}
        bank_seen: dict[int, float] = {}
        hp_by_turn: dict[int, dict[str, float]] = {}
        targets: dict[str, dict[int, list[tuple[str, str]]]] = {"mine": {}, "theirs": {}}
        faints: dict[int, list[str]] = {}
        fainted_this_turn: dict[str, bool] = {"mine": False, "theirs": False}
        voluntary: list[int] = []
        protect_attempts = repeat_protects = 0
        for raw in log:
            parts = str(raw).split("|")
            if len(parts) >= 3 and parts[1] == "turn":
                try:
                    turn = int(parts[2])
                except ValueError:
                    pass
                fainted_this_turn = {"mine": False, "theirs": False}
                continue
            if len(parts) < 3:
                continue
            kind, actor = parts[1], parts[2]
            if kind == "inactive":
                text = "|".join(parts[2:]).strip()
                m = re.match(r"Time left: (\d+) sec this turn \| (\d+) sec total", text)
                if m:
                    bank_seen[turn] = float(m.group(2))  # sent to us about our own clock
                m = re.match(r"(.+?) has (\d+) seconds left\.", text)
                if m:
                    whose = next((tag for tag, name in players.items() if name and m.group(1).strip() == name), None)
                    if whose:
                        left = int(m.group(2))
                        slow[whose][turn] = min(slow[whose].get(turn, left), left)
                continue
            side = actor.split(":")[0].rstrip("ab")
            if not side.startswith("p"):
                continue
            ours = side == my_side
            tag_side = "mine" if ours else "theirs"
            if kind == "player" and len(parts) > 3:
                players[tag_side] = parts[3].strip()
                continue
            if kind == "poke" and len(parts) > 3:
                name = parts[3].split(",")[0].strip()
                if name and name not in rosters[tag_side]:
                    rosters[tag_side].append(name)
                continue
            if kind in ("-sidestart", "-sideend") and len(parts) > 3:
                # "|-sidestart|p1: name|Spikes" (hazards) / "|-sidestart|p2: name|move: Tailwind" (the rest)
                effect = species_key(parts[3].split(":")[-1])
                table = hazards[tag_side]
                if effect in ("spikes", "toxicspikes"):
                    table[effect] = min(table.get(effect, 0) + 1, 3 if effect == "spikes" else 2) if kind == "-sidestart" else 0
                elif effect in ("stealthrock", "stickyweb"):
                    table[effect] = kind == "-sidestart"
                continue
            nick = (side, species_key(actor.split(":", 1)[1].strip() if ":" in actor else actor))
            if kind in ("switch", "drag", "replace") and len(parts) > 3:
                species = species_key(parts[3].split(",")[0])
                nick_to_species[nick] = species
                if kind != "replace":  # Illusion breaking is not an entry: the Pokémon has been acting all along
                    unburden.discard(tag_side + ":" + species)  # a new stint: Unburden only counts items lost on the field
                    entered.setdefault(turn, []).append((tag_side, species))
                    acted[tag_side + ":" + species] = False
                    if kind == "switch" and turn >= 1 and not ours and "[from]" not in str(raw) and not fainted_this_turn["theirs"]:
                        voluntary.append(turn)  # their own choice: not a replacement, not U-turn/Volt Switch/Eject
                    hp = _hp_fraction(parts[4] if len(parts) > 4 else "")
                    if hp is not None:
                        hp_by_turn.setdefault(turn, {})[tag_side + ":" + species] = hp
                if not ours:
                    last[species] = None  # fresh on the field: no lock yet
                continue
            species = nick_to_species.get(nick, nick[1])
            tag = tag_side + ":" + species
            if kind in ("move", "cant"):
                acted[tag] = True  # a move action, even one that flinched or failed: Fake Out's first turn is spent
            if kind in ("-damage", "-heal", "-sethp") and len(parts) > 3:
                hp = _hp_fraction(parts[3])
                if hp is not None:
                    hp_by_turn.setdefault(turn, {})[tag] = hp
            if kind == "move" and len(parts) > 4 and ":" in parts[4]:
                target_nick = (parts[4].split(":")[0].rstrip("ab"), species_key(parts[4].split(":", 1)[1].strip()))
                targets[tag_side].setdefault(turn, []).append((species, nick_to_species.get(target_nick, target_nick[1])))
            if kind == "move" and len(parts) > 3 and not ours and species_key(parts[3]) in self.PROTECT_LIKE:
                protect_attempts += 1
                if moves_by_turn["theirs"].get(species, {}).get(turn - 1) in self.PROTECT_LIKE:
                    repeat_protects += 1
            if kind == "-start" and len(parts) > 3:
                effect = species_key(parts[3])
                for ability in ("protosynthesis", "quarkdrive"):
                    if effect.startswith(ability) and len(effect) > len(ability):
                        paradox[tag] = effect[len(ability):]  # "atk", "def", "spa", "spd", "spe"
                if effect.startswith("fallen") and effect[6:].isdigit():
                    overlord[tag] = int(effect[6:])  # Supreme Overlord announced on entry: "|-start|p2a: Kingambit|fallen2|[silent]"
            elif kind == "-end" and len(parts) > 3 and species_key(parts[3]) in ("protosynthesis", "quarkdrive"):
                paradox.pop(tag, None)
            elif kind == "-end" and len(parts) > 3 and species_key(parts[3]).startswith("fallen"):
                overlord.pop(tag, None)
            elif kind == "faint":
                paradox.pop(tag, None)
                unburden.discard(tag)
                overlord.pop(tag, None)
                fallen[tag_side] += 1
                fainted_this_turn[tag_side] = True
                faints.setdefault(turn, []).append(tag)
                hp_by_turn.setdefault(turn, {})[tag] = 0.0
            if kind == "-enditem":
                unburden.add(tag)  # the item is gone while it stands on the field: Unburden doubles its speed
            if kind == "move" and len(parts) > 3:
                move_id = species_key(parts[3])
                moves_by_turn[tag_side].setdefault(species, {})[turn] = move_id
                # A Protect that fails is announced "|move|X|Protect||[still]" and followed by "|-fail|X".
                pending_protect = (tag_side, species, turn) if move_id in self.PROTECT_LIKE else None
            elif kind == "-fail" and pending_protect and pending_protect[:2] == (tag_side, species):
                failed_protect[tag_side].setdefault(species, set()).add(pending_protect[2])
                pending_protect = None
            if ours:
                continue  # below: what we track for the opponent only
            if kind == "move" and len(parts) > 3:
                last[species] = species_key(parts[3])
            elif kind == "-enditem":
                lost.add(species)  # popped Air Balloon, eaten berry, used Booster Energy, Knock Off
            elif kind == "-item":
                lost.discard(species)  # revealed (or received by Trick): it holds one now
        self.opp_last_move = last
        self.opp_items_lost = sorted(lost)
        self.paradox_active = paradox
        self.unburden_active = sorted(unburden)
        self.moves_by_turn = moves_by_turn
        self.entered_by_turn = entered
        self.acted_since_entry = acted
        self.roster_by_side = rosters
        self.protect_failed = failed_protect
        self.fallen_by_side = fallen
        self.overlord_fallen = overlord
        self.hazards_by_side = hazards
        self.players = players
        self.their_slow_turns = slow["theirs"]
        self.my_slow_turns = slow["mine"]
        self.my_bank_seen = bank_seen
        self.hp_by_turn = hp_by_turn
        self.targets_by_turn = targets
        self.faints_by_turn = faints
        self.their_voluntary_switches = voluntary
        self.their_protect_attempts = protect_attempts
        self.their_repeat_protects = repeat_protects
        self.log_turn = turn
        # Consecutive Protects by each opponent, ending with the turn that just finished (read from the log, not inferred).
        streaks: dict[str, int] = {}
        for species in moves_by_turn["theirs"]:
            n = self.logged_streak("theirs", species)
            if n:
                streaks[species] = n
        self.opp_protect_streak = streaks
        # With a log, "who Protected last turn" is a fact, not an inference from an HP bar that did not move (which
        # also fires after a miss, an immunity, a flinch or a Spore).
        self.opp_protected_last_turn = sorted(streaks)

    def observe_opponent_hp(self, obs: dict) -> None:
        """Infer who Protected: an opponent we targeted last turn whose HP did not change. Only without a protocol
        log (``observe_protocol`` reads the executed Protect itself when there is one)."""
        current: dict[str, float] = {}
        for entry in obs.get("opponent_active_pokemon") or []:
            if isinstance(entry, dict) and entry.get("species") is not None:
                frac = entry.get("current_hp_fraction")
                if frac is None and entry.get("max_hp"):
                    frac = (entry.get("current_hp") or 0) / entry["max_hp"]
                if frac is not None:
                    current[species_key(entry["species"])] = float(frac)
        targeted = set()
        if self.last_payload and not obs.get("protocol_log"):
            for slot_key in ("slot_0", "slot_1"):
                choice = self.last_payload.get(slot_key) or {}
                tgt = choice.get("target")
                if choice.get("type") != "move" or not isinstance(tgt, int) or tgt <= 0 or len(self.opp_last_active) < tgt:
                    continue
                info = data.move_info(choice.get("move_id")) or {}
                if not (info.get("base_power") or 0) > 0:
                    continue  # Spore, Taunt, Tailwind...: an untouched HP bar says nothing about Protect
                target = self.opp_last_active[tgt - 1]
                if not target:
                    continue
                card_moves = {species_key(m.get("id") if isinstance(m, dict) else m) for m in (self.opp_cards.get(target) or {}).get("moves") or []}
                if card_moves & self.PROTECT_LIKE:  # only a set that has Protect can have Protected (L2 T2: Incineroar had none)
                    targeted.add(target)
        self.opp_protected_last_turn = [k for k in targeted if k in current and k in self.last_opp_hp and abs(current[k] - self.last_opp_hp[k]) < 0.005]
        self.last_opp_hp = current

    # -- prompt material -------------------------------------------------------------

    def known_sets(self, side: str, *, only: list[str] | None = None, status: dict[str, str] | None = None) -> list[dict]:
        """Compact card summaries for the prompt: what we know for sure. ``only`` keeps just those species (our four in
        this battle); ``status`` tags each card with where it is (on_field / in_back / fainted / NOT_IN_THIS_BATTLE).
        The item shown is the drafted one: in battle, the turn sheet knows when it has been used up or knocked off."""
        cards = self.my_cards if side == "mine" else self.opp_cards
        keep = {species_key(s) for s in only} if only else None
        out = []
        for key, card in cards.items():
            if keep is not None and key not in keep:
                continue
            entry = {
                "species": card.get("species"),
                "item": card.get("item"),
                "ability": card.get("ability"),
                "nature": card.get("nature"),
                "types": card.get("types"),
                "base_stats": card.get("base_stats"),
                "evs": card.get("evs"),
                "moves": card.get("moves"),
            }
            if status is not None:
                entry["status"] = status.get(key, "unknown")
            out.append(entry)
        return out

    def summary(self) -> dict:
        return {
            "my_cards": len(self.my_cards),
            "opp_cards": len(self.opp_cards),
            "pool_cards": len(self.pool_cards),
            "turns": len(self.turns),
            "llm_calls": self.llm_calls,
            "fallbacks": self.fallbacks,
        }
