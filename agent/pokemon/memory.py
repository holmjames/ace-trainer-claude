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
        actives = []
        for entry in obs.get("active_pokemon") or []:
            key = species_key(entry.get("species") if isinstance(entry, dict) else entry)
            if key:
                actives.append(key)
        if actives:
            self.fresh_active = self._fresh("mine", actives, self.last_active, turn)
            self.last_active = actives
        opp_actives = []
        for entry in obs.get("opponent_active_pokemon") or []:
            key = species_key(entry.get("species") if isinstance(entry, dict) else entry)
            if key:
                opp_actives.append(key)
        if opp_actives:
            self.opp_fresh_active = self._fresh("theirs", opp_actives, self.opp_last_active, turn)
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
        brought.update(self.last_active)
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

    def record_turn(self, **fields: Any) -> None:
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
        by_turn = (self.moves_by_turn.get(side) or {}).get(species_key(species)) or {}
        n, t = 0, (before_turn if before_turn is not None else self.log_turn) - 1
        while by_turn.get(t) in self.PROTECT_LIKE:
            n, t = n + 1, t - 1
        return n

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
        for raw in log:
            parts = str(raw).split("|")
            if len(parts) >= 3 and parts[1] == "turn":
                try:
                    turn = int(parts[2])
                except ValueError:
                    pass
                continue
            if len(parts) < 3:
                continue
            kind, actor = parts[1], parts[2]
            side = actor.split(":")[0].rstrip("ab")
            if not side.startswith("p"):
                continue
            ours = side == my_side
            tag_side = "mine" if ours else "theirs"
            if kind == "poke" and len(parts) > 3:
                name = parts[3].split(",")[0].strip()
                if name and name not in rosters[tag_side]:
                    rosters[tag_side].append(name)
                continue
            nick = (side, species_key(actor.split(":", 1)[1].strip() if ":" in actor else actor))
            if kind in ("switch", "drag", "replace") and len(parts) > 3:
                species = species_key(parts[3].split(",")[0])
                nick_to_species[nick] = species
                if kind != "replace":  # Illusion breaking is not an entry: the Pokémon has been acting all along
                    unburden.discard(tag_side + ":" + species)  # a new stint: Unburden only counts items lost on the field
                    entered.setdefault(turn, []).append((tag_side, species))
                    acted[tag_side + ":" + species] = False
                if not ours:
                    last[species] = None  # fresh on the field: no lock yet
                continue
            species = nick_to_species.get(nick, nick[1])
            tag = tag_side + ":" + species
            if kind in ("move", "cant"):
                acted[tag] = True  # a move action, even one that flinched or failed: Fake Out's first turn is spent
            if kind == "-start" and len(parts) > 3:
                effect = species_key(parts[3])
                for ability in ("protosynthesis", "quarkdrive"):
                    if effect.startswith(ability) and len(effect) > len(ability):
                        paradox[tag] = effect[len(ability):]  # "atk", "def", "spa", "spd", "spe"
            elif kind == "-end" and len(parts) > 3 and species_key(parts[3]) in ("protosynthesis", "quarkdrive"):
                paradox.pop(tag, None)
            elif kind == "faint":
                paradox.pop(tag, None)
                unburden.discard(tag)
            if kind == "-enditem":
                unburden.add(tag)  # the item is gone while it stands on the field: Unburden doubles its speed
            if kind == "move" and len(parts) > 3:
                moves_by_turn[tag_side].setdefault(species, {})[turn] = species_key(parts[3])
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
                if choice.get("type") == "move" and isinstance(tgt, int) and tgt > 0 and len(self.opp_last_active) >= tgt:
                    targeted.add(self.opp_last_active[tgt - 1])
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
