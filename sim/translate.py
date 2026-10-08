"""Turn local-engine snapshots (sim/bridge.js) into exactly the payloads the live tournament server sends.

Plain language: the live server runs Pokémon Showdown behind a poke-env style client and hands our agent a
JSON state: an observation plus ``legal_actions``. Every field of that state is reproduced here, field for
field, from the same two things a live client receives — its own view of the Showdown protocol log and its
own Showdown requests — so offline games exercise the same agent code paths as real matches, including the
awkward conventions. The ground truth is the captured live payloads in ``tests/fixtures/live/``;
``tests/test_sim_live_parity.py`` fails loudly the moment this drifts from them.

Live conventions reproduced (each one hid a bug at some point):

* Spread / self / field moves carry ``"targets": [0]`` and ``target_options`` ``[{"target": 0, "side": "none",
  "species": null}]``; single-target moves list the ally position too (``[-2, 1, 2]`` for slot 0), and every
  position stays listed even when nobody stands there (``species: null``).
* Team Preview ``opponent_roster`` entries are blank cards: ``moves: []``, ``item: "unknown_item"``,
  ``current_hp: 0``, ``max_hp: 0``; ``ability`` is only filled when the species has a single possible ability.
* In battle, ``opponent_team`` lists only opponents that have appeared. ``item`` is ``"unknown_item"`` until
  the log reveals it, the item id once revealed, ``null`` once consumed or removed; ``ability`` is ``null``
  until revealed; ``moves`` are only the moves it has used; HP is a percentage (``max_hp`` 100).
* ``team`` lists all six drafted Pokémon (the two left at home keep the Team Preview request's ``active``
  flag, so one can read ``active: true`` forever); a consumed item reads ``""``.
* ``weather`` / ``fields`` / ``side_conditions`` are dicts keyed by UPPER_SNAKE names whose value is the turn
  the condition started (layers for Spikes); ``protocol_log`` is the full cumulative log.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from altruagent.models import GameState

from agent.pokemon import data

GAME_TYPE = "pokemon_vgc_doubles_draft"
BATTLE_FORMAT = "gen9vgc2025regi"
UNKNOWN_ITEM = "unknown_item"
BOOST_KEYS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
STATUS_NAMES = {"brn": "BRN", "frz": "FRZ", "par": "PAR", "psn": "PSN", "slp": "SLP", "tox": "TOX", "fnt": "FNT"}
STACKABLE_SIDE_CONDITIONS = frozenset({"SPIKES", "TOXIC_SPIKES"})
NOT_STORED_MOVES = frozenset({"struggle", "recharge"})
DRAFT_PICKS = 12
DRAFT_TIMEOUT_SECONDS = 15.0
PREVIEW_CLOCK = {"decision_seconds_left": 85, "bank_seconds_left": 420}
BATTLE_CLOCK = {"decision_seconds_left": 50, "bank_seconds_left": 420}
NEXT_ACTIONS = [{"tool": "play_action", "hint": "Pick one of legal_actions.actions and pass its `input`."}]

TARGET_LEGEND = {
    "-1": "board position A on your side (always slot 0's position)",
    "-2": "board position B on your side (always slot 1's position)",
    "1": "board position A on the opponent's side",
    "2": "board position B on the opponent's side",
    "0": "no target needed (self/field/spread move)",
}
DOUBLES_LABEL = "Choose actions for both active Pokémon"
DOUBLES_INSTRUCTIONS = (
    "Submit one action for slot 0 and one for slot 1 together via play_action with "
    "action={'type':'doubles_turn','slot_0':{...},'slot_1':{...}}. Each slot is "
    "{'type':'move','move_id':...,'target': <int>} (target required unless the move's only legal target is 0), "
    "{'type':'switch','species':...}, or {'type':'pass'} (required when this slot's own options list is just "
    "[{'type':'pass'}]). IMPORTANT: -1 and -2 are FIXED board positions (see target_legend), not 'self'/'ally' "
    "labels -- check each slot's own board_position to know which one it is. For slot 0 (board_position -1), -1 "
    "means targeting itself and -2 means its ally. For slot 1 (board_position -2), it's the reverse: -2 is itself "
    "and -1 is its ally. Never assume -1 always means self. Simplest: read each move's target_options, which names "
    "every legal target for that slot as side 'opponent', 'ally' or 'self' with the species standing there now -- "
    "then submit that option's integer as 'target'. Targeting your own ally is legal but hits your own Pokémon."
)
LINEUP_LABEL = "Choose 4 of 6 and designate 2 leads"
LINEUP_INSTRUCTIONS = (
    "Choose exactly 4 species from `roster` to bring into battle, and which 2 of those 4 lead. Submit via "
    "play_action with action={'type': 'select_lineup', 'bring': [4 species from roster], 'leads': [2 of those 4 species]}."
)


# -- small helpers -------------------------------------------------------------------------------------


def _upper_snake(name: str) -> str:
    """Showdown camelCase target -> poke-env enum name: ``allAdjacentFoes`` -> ``ALL_ADJACENT_FOES``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(name)).upper()


def _weather_name(text: str) -> str:
    """``RainDance`` -> ``RAINDANCE`` (poke-env ``Weather``)."""
    return re.sub(r"[ \-]", "", text.replace("move: ", "")).upper()


def _field_name(text: str) -> str:
    """``move: Grassy Terrain`` -> ``GRASSY_TERRAIN``, ``move: Trick Room`` -> ``TRICK_ROOM`` (poke-env ``Field``)."""
    name = text.replace("move: ", "").strip().replace(" ", "_")
    if name.lower().endswith("terrain") and not name.lower().endswith("_terrain"):
        name = name[: -len("terrain")] + "_terrain"
    return name.upper()


def _side_condition_name(text: str) -> str:
    """``move: Tailwind`` -> ``TAILWIND``, ``Light Screen`` -> ``LIGHT_SCREEN`` (poke-env ``SideCondition``)."""
    return text.replace("move: ", "").strip().replace(" ", "_").replace("-", "_").upper()


def _accuracy(value: Any) -> float:
    if value is True or value is None:
        return 1.0
    return float(value) / 100


def _effect_value(part: str, prefix: str) -> str | None:
    """``[from] item: Life Orb`` -> ``Life Orb`` for prefix ``item:`` (with or without the space after ``[from]``)."""
    text = part.replace("[from]", "", 1).strip() if part.startswith("[from]") else None
    if text is None or not text.startswith(prefix):
        return None
    return text[len(prefix):].strip()


def _new_state(session_id: str, phase: str, observation: dict, legal: list[dict], version: int, *, actor: str,
               position: int, my_turn: bool = True) -> GameState:
    """The full server state dict, key for key as the live server sends it."""
    return GameState.from_mcp_state({
        "session_id": session_id,
        "game_type": GAME_TYPE,
        "runtime_adapter": "pokemon",
        "status": "in_progress",
        "state_version": version,
        "observation": observation,
        "phase": phase,
        "is_terminal": False,
        "is_current_actor": my_turn,
        "current_actor": {"agent_id": actor, "position": position},
        "legal_actions": {"session_id": session_id, "state_version": version, "actions": legal},
        "next_actions": copy.deepcopy(NEXT_ACTIONS),
        "updated": True,
    })


def _action(session_id: str, action_id: str, label: str, action: dict, version: int) -> dict:
    return {"action_id": action_id, "label": label,
            "input": {"session_id": session_id, "action": action, "action_id": action_id, "state_version": version}}


# -- one Pokémon, as a poke-env client tracks it -------------------------------------------------------


class Mon:
    """A Pokémon as the live server's client sees it; ``to_obs`` is the exact dict the server sends."""

    def __init__(self, species: str, name: str | None = None) -> None:
        self.species = ""
        self.name = name or str(species)
        self.current_hp = 0
        self.max_hp = 0
        self.status: str | None = None
        self.ability: str | None = None
        self.item: str | None = UNKNOWN_ITEM
        self.types: list[str] = []
        self.base_stats: dict[str, int] = {}
        self.boosts = dict.fromkeys(BOOST_KEYS, 0)
        self.moves: list[str] = []
        self.active = False
        self.revealed = False
        self.is_terastallized = False
        self.set_species(species)

    # identity
    def set_species(self, species: str) -> None:
        species_id = data.to_id(species)
        if species_id == self.species:
            return
        self.species = species_id
        info = data.species_info(species_id) or {}
        self.types = [str(t).upper() for t in info.get("types") or []]
        self.base_stats = dict(info.get("base_stats") or {})
        abilities = [data.to_id(a) for a in info.get("abilities") or []]
        if len(abilities) == 1:  # a species with one possible ability shows it before it is revealed
            self.ability = abilities[0]

    def update_details(self, details: str | None) -> None:
        if not details:
            return
        parts = [p.strip() for p in details.split(",") if not p.strip().startswith("tera:") and p.strip() != "shiny"]
        if parts:
            self.set_species(parts[0])

    # state changes
    @property
    def fainted(self) -> bool:
        return self.status == "FNT"

    def set_hp_status(self, text: str | None) -> None:
        if not text:
            return
        hp, _, status = text.strip().partition(" ")
        if status.strip().lower() == "fnt":  # "0 fnt"
            self.faint()
            return
        if status:
            self.status = STATUS_NAMES.get(status.strip().lower(), self.status)
        nums = "".join(c for c in hp if c in "0123456789/").split("/")
        if len(nums) == 2 and nums[0] and nums[1]:
            self.current_hp, self.max_hp = int(nums[0]), int(nums[1])

    def faint(self) -> None:
        self.current_hp = 0
        self.status = "FNT"

    def cure_status(self, status: str | None = None) -> None:
        if status and STATUS_NAMES.get(status.strip().lower()) == self.status:
            self.status = None
        elif not status and not self.fainted:
            self.status = None

    def boost(self, stat: str, amount: int) -> None:
        if stat in self.boosts:
            self.boosts[stat] = max(-6, min(6, self.boosts[stat] + amount))

    def clear_boosts(self) -> None:
        self.boosts = dict.fromkeys(BOOST_KEYS, 0)

    def add_move(self, move: str) -> None:
        move_id = data.to_id(move)
        if move_id and move_id not in NOT_STORED_MOVES and move_id not in self.moves:
            self.moves.append(move_id)

    def switch_in(self, details: str | None = None) -> None:
        self.active = True
        self.update_details(details)
        self.revealed = True

    def switch_out(self) -> None:
        self.active = False
        self.clear_boosts()

    def update_from_request(self, entry: dict) -> None:
        """One ``side.pokemon`` entry of our own Showdown request (exact HP, item, ability and moves)."""
        self.active = bool(entry.get("active"))
        ability = entry.get("ability") or entry.get("baseAbility")
        if ability:
            self.ability = ability
        self.set_hp_status(entry.get("condition"))
        self.item = entry.get("item", "")
        self.update_details(entry.get("details"))
        for move in entry.get("moves") or []:
            self.add_move(move)
        wanted = {data.to_id(m) for m in entry.get("moves") or []}
        if len(self.moves) > 4:
            self.moves = [m for m in self.moves if m in wanted]
        if entry.get("terastallized"):
            self.terastallize(entry["terastallized"])

    def terastallize(self, tera_type: str) -> None:
        self.is_terastallized = True
        if tera_type:
            self.types = [str(tera_type).upper()]

    def to_obs(self) -> dict:
        return {
            "species": self.species,
            "name": self.name,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "current_hp_fraction": self.current_hp / self.max_hp if self.current_hp and self.max_hp else 0,
            "status": self.status,
            "ability": self.ability,
            "item": self.item,
            "types": list(self.types),
            "base_stats": dict(self.base_stats),
            "boosts": dict(self.boosts),
            "moves": list(self.moves),
            "fainted": self.fainted,
            "active": self.active,
            "revealed": self.revealed,
            "is_terastallized": self.is_terastallized,
        }


# -- one player's view of a battle: protocol log + its own requests -------------------------------------


class BattleView:
    """What one seat's client knows: built from that seat's protocol log and its own Showdown requests."""

    def __init__(self, role: str, *, base_species: dict[str, str] | None = None) -> None:
        self.role = role
        self.opp_role = "p2" if role == "p1" else "p1"
        self.base_species = dict(base_species or {})  # species id -> base species name (Team Preview names)
        self.team: dict[str, Mon] = {}
        self.opponent_team: dict[str, Mon] = {}
        self.teampreview_opponent: list[Mon] = []
        self.active: dict[str, Mon] = {}  # "p1a" -> Pokémon in that position (fainted ones stay until replaced)
        self.opponent_active: dict[str, Mon] = {}
        self.turn = 0
        self.weather: dict[str, int] = {}
        self.fields: dict[str, int] = {}
        self.side_conditions: dict[str, int] = {}
        self.opponent_side_conditions: dict[str, int] = {}
        self.player_names: dict[str, str] = {}
        self.finished = False
        self.won: bool | None = None
        self.lost: bool | None = None
        self.force_switch = [False, False]
        self.available_moves: list[list[dict]] = [[], []]
        self.available_switches: list[list[Mon]] = [[], []]
        self.can_tera = [False, False]
        self.trapped = [False, False]

    # lookups
    def get_pokemon(self, identifier: str, details: str | None = None) -> Mon:
        side, _, name = str(identifier).partition(": ")
        role, name = side[:2], name.strip()
        team = self.team if role == self.role else self.opponent_team
        key = f"{role}: {name}"
        mon = team.get(key)
        if mon is None:
            species = details.split(",")[0] if details else name
            mon = Mon(species, name=name)
            team[key] = mon
        if details:
            mon.update_details(details)
        return mon

    def _actives(self, role: str) -> dict[str, Mon]:
        return self.active if role == self.role else self.opponent_active

    @property
    def active_pokemon(self) -> list[Mon | None]:
        return [self._slot(self.active, f"{self.role}{p}") for p in "ab"]

    @property
    def opponent_active_pokemon(self) -> list[Mon | None]:
        return [self._slot(self.opponent_active, f"{self.opp_role}{p}") for p in "ab"]

    @staticmethod
    def _slot(actives: dict[str, Mon], key: str) -> Mon | None:
        mon = actives.get(key)
        return mon if mon is not None and mon.active and not mon.fainted else None

    # the protocol log
    def feed_all(self, lines: Iterable[str]) -> "BattleView":
        for line in lines:
            self.feed(line)
        return self

    def feed(self, line: str) -> None:
        parts = str(line).split("|")
        if len(parts) < 2 or not parts[1]:
            return
        handler = getattr(self, "_on_" + parts[1].lstrip("-").replace("-", "_"), None)
        if handler is None:
            return
        if len(parts) < 3 and parts[1] not in ("-clearallboost", "tie"):
            return
        handler(parts)

    def _on_player(self, p: list[str]) -> None:
        if len(p) > 3 and p[3]:
            self.player_names[p[2]] = p[3]

    def _on_poke(self, p: list[str]) -> None:
        if p[2] != self.role and len(p) > 3 and p[3]:
            species_name = p[3].split(",")[0].strip()
            species = data.to_id(species_name)
            self.teampreview_opponent.append(Mon(species, name=self.base_species.get(species) or species_name))

    def _on_switch(self, p: list[str]) -> None:
        ident = p[2]
        position = ident.split(":")[0].strip()
        actives = self._actives(position[:2])
        out = actives.pop(position, None)
        if out is not None:
            out.switch_out()
        mon = self.get_pokemon(ident, p[3] if len(p) > 3 else None)
        mon.switch_in(p[3] if len(p) > 3 else None)
        if len(p) > 4:
            mon.set_hp_status(p[4])
        actives[position] = mon

    _on_drag = _on_switch

    def _on_swap(self, p: list[str]) -> None:
        role = p[2][:2]
        actives = self._actives(role)
        a, b = actives.get(f"{role}a"), actives.get(f"{role}b")
        if a is None or b is None or a.fainted or b.fainted:
            return
        actives[f"{role}a"], actives[f"{role}b"] = b, a

    def _on_detailschange(self, p: list[str]) -> None:
        if len(p) > 3:
            self.get_pokemon(p[2]).update_details(p[3])

    def _on_formechange(self, p: list[str]) -> None:
        if len(p) > 3:
            mon = self.get_pokemon(p[2])
            info = data.species_info(p[3]) or {}
            if info:
                mon.types = [str(t).upper() for t in info.get("types") or []]
                mon.base_stats = dict(info.get("base_stats") or {})

    def _on_move(self, p: list[str]) -> None:
        if len(p) < 4:
            return
        mon = self.get_pokemon(p[2])
        extras = p[4:]
        override = None
        for part in extras:
            ability = _effect_value(part, "ability:")
            if ability:
                mon.ability = data.to_id(ability)
                if data.to_id(ability) in ("magicbounce", "dancer"):
                    return  # a reflected or copied move is not one of its own
            called_by = _effect_value(part, "move:")
            if called_by:
                override = called_by
        mon.add_move(override or p[3])

    def _on_damage(self, p: list[str]) -> None:
        if len(p) < 4:
            return
        mon = self.get_pokemon(p[2])
        mon.set_hp_status(p[3])
        if len(p) == 6 and p[5].startswith("[of]"):
            source = self.get_pokemon(p[5][len("[of]"):].strip())
            item = _effect_value(p[4], "item:")
            ability = _effect_value(p[4], "ability:")
            if item:
                source.item = data.to_id(item)  # Rocky Helmet: the item belongs to the [of] Pokémon
            if ability:
                source.ability = data.to_id(ability)  # Rough Skin
        elif len(p) == 5:
            item = _effect_value(p[4], "item:")
            if item:
                mon.item = data.to_id(item)  # Life Orb recoil

    def _on_heal(self, p: list[str]) -> None:
        if len(p) < 4:
            return
        mon = self.get_pokemon(p[2])
        mon.set_hp_status(p[3])
        if len(p) == 5:
            item = _effect_value(p[4], "item:")
            if item and mon.item is not None:  # a berry eaten a moment ago stays gone
                mon.item = data.to_id(item)  # Leftovers
        if len(p) == 6:
            ability = _effect_value(p[4], "ability:")
            if ability:
                mon.ability = data.to_id(ability)  # Water Absorb and friends

    def _on_sethp(self, p: list[str]) -> None:
        if len(p) > 3:
            self.get_pokemon(p[2]).set_hp_status(p[3])

    def _on_faint(self, p: list[str]) -> None:
        self.get_pokemon(p[2]).faint()

    def _on_status(self, p: list[str]) -> None:
        if len(p) > 3:
            mon = self.get_pokemon(p[2])
            mon.status = STATUS_NAMES.get(p[3].strip().lower(), mon.status)

    def _on_curestatus(self, p: list[str]) -> None:
        self.get_pokemon(p[2]).cure_status(p[3] if len(p) > 3 else None)

    def _on_cureteam(self, p: list[str]) -> None:
        role = p[2][:2]
        for mon in (self.team if role == self.role else self.opponent_team).values():
            mon.cure_status()

    def _on_boost(self, p: list[str]) -> None:
        if len(p) > 4 and p[4].lstrip("-").isdigit():
            self.get_pokemon(p[2]).boost(p[3], int(p[4]))

    def _on_unboost(self, p: list[str]) -> None:
        if len(p) > 4 and p[4].lstrip("-").isdigit():
            self.get_pokemon(p[2]).boost(p[3], -int(p[4]))

    def _on_setboost(self, p: list[str]) -> None:
        if len(p) > 4 and p[4].lstrip("-").isdigit():
            mon = self.get_pokemon(p[2])
            if p[3] in mon.boosts:
                mon.boosts[p[3]] = max(-6, min(6, int(p[4])))

    def _on_clearboost(self, p: list[str]) -> None:
        self.get_pokemon(p[2]).clear_boosts()

    def _on_clearallboost(self, p: list[str]) -> None:
        for mon in list(self.active.values()) + list(self.opponent_active.values()):
            mon.clear_boosts()

    def _on_clearnegativeboost(self, p: list[str]) -> None:
        mon = self.get_pokemon(p[2])
        mon.boosts = {k: max(0, v) for k, v in mon.boosts.items()}

    def _on_clearpositiveboost(self, p: list[str]) -> None:
        mon = self.get_pokemon(p[2])
        mon.boosts = {k: min(0, v) for k, v in mon.boosts.items()}

    def _on_invertboost(self, p: list[str]) -> None:
        mon = self.get_pokemon(p[2])
        mon.boosts = {k: -v for k, v in mon.boosts.items()}

    def _on_copyboost(self, p: list[str]) -> None:
        if len(p) > 3:  # |-copyboost|SOURCE|TARGET: SOURCE copies TARGET's stat changes (Psych Up)
            self.get_pokemon(p[2]).boosts = dict(self.get_pokemon(p[3]).boosts)

    def _on_swapboost(self, p: list[str]) -> None:
        if len(p) > 3:
            a, b = self.get_pokemon(p[2]), self.get_pokemon(p[3])
            stats = [s.strip() for s in p[4].split(",")] if len(p) > 4 and p[4] and not p[4].startswith("[") else list(BOOST_KEYS)
            for stat in stats:
                if stat in a.boosts:
                    a.boosts[stat], b.boosts[stat] = b.boosts[stat], a.boosts[stat]

    def _on_item(self, p: list[str]) -> None:
        if len(p) > 3:
            self.get_pokemon(p[2]).item = data.to_id(p[3])  # Air Balloon on switch-in, Frisk, Trick...

    def _on_enditem(self, p: list[str]) -> None:
        self.get_pokemon(p[2]).item = None  # eaten, popped, knocked off, used up

    def _on_ability(self, p: list[str]) -> None:
        if len(p) > 3:
            if len(p) > 4 and _effect_value(p[4], "move:"):
                return  # a temporary ability (Skill Swap and friends)
            self.get_pokemon(p[2]).ability = data.to_id(p[3])

    def _on_terastallize(self, p: list[str]) -> None:
        self.get_pokemon(p[2]).terastallize(p[3] if len(p) > 3 else "")

    def _on_weather(self, p: list[str]) -> None:
        if p[2] == "none":
            self.weather = {}
        else:
            self.weather = {_weather_name(p[2]): self.turn}

    def _on_fieldstart(self, p: list[str]) -> None:
        field = _field_name(p[2])
        if field.endswith("_TERRAIN"):
            self.fields = {f: t for f, t in self.fields.items() if not f.endswith("_TERRAIN")}
        self.fields[field] = self.turn

    def _on_fieldend(self, p: list[str]) -> None:
        self.fields.pop(_field_name(p[2]), None)

    def _side_dict(self, side: str) -> dict[str, int]:
        return self.side_conditions if side[:2] == self.role else self.opponent_side_conditions

    def _on_sidestart(self, p: list[str]) -> None:
        if len(p) > 3:
            conditions = self._side_dict(p[2])
            name = _side_condition_name(p[3])
            if name in STACKABLE_SIDE_CONDITIONS:
                conditions[name] = conditions.get(name, 0) + 1
            elif name not in conditions:
                conditions[name] = self.turn

    def _on_sideend(self, p: list[str]) -> None:
        if len(p) > 3:
            self._side_dict(p[2]).pop(_side_condition_name(p[3]), None)

    def _on_swapsideconditions(self, p: list[str]) -> None:
        self.side_conditions, self.opponent_side_conditions = self.opponent_side_conditions, self.side_conditions

    def _on_turn(self, p: list[str]) -> None:
        if p[2].strip().isdigit():
            self.turn = int(p[2])

    def _on_win(self, p: list[str]) -> None:
        self.finished = True
        self.won = p[2] == self.player_names.get(self.role)
        self.lost = not self.won

    def _on_tie(self, p: list[str]) -> None:
        self.finished = True
        self.won, self.lost = False, False

    # our own Showdown requests
    def apply_request(self, request: dict, *, current: bool = True) -> None:
        side = request.get("side") or {}
        entries = [e for e in side.get("pokemon") or [] if isinstance(e, dict) and e.get("ident")]
        for entry in entries:
            self.get_pokemon(entry["ident"], entry.get("details")).update_from_request(entry)
        if not current:
            return
        force = [bool(f) for f in (request.get("forceSwitch") or [])][:2]
        self.force_switch = force + [False] * (2 - len(force))
        self.available_moves, self.can_tera, self.trapped = [[], []], [False, False], [False, False]
        for i, active in enumerate((request.get("active") or [])[:2]):
            if i >= len(entries) or not isinstance(active, dict):
                continue
            if self.get_pokemon(entries[i]["ident"]).fainted:
                continue
            self.available_moves[i] = [m for m in active.get("moves") or [] if not m.get("disabled")]
            self.trapped[i] = bool(active.get("trapped"))
            self.can_tera[i] = bool(active.get("canTerastallize"))
        self.available_switches = [[], []]
        if request.get("teamPreview"):
            return
        for i in range(2):
            if self.trapped[i]:
                continue
            for entry in entries:
                mon = self.get_pokemon(entry["ident"])
                if not mon.active and not mon.fainted:
                    self.available_switches[i].append(mon)

    # moves and their targets
    def target_options(self, targets: list[int], slot: int) -> list[dict]:
        mine, theirs = self.active_pokemon, self.opponent_active_pokemon
        out = []
        for target in targets:
            if target == 0:
                out.append({"target": 0, "side": "none", "species": None})
            elif target > 0:
                mon = theirs[target - 1] if target <= 2 else None
                out.append({"target": target, "side": "opponent", "species": mon.species if mon else None})
            else:
                mon = mine[-target - 1] if -target <= 2 else None
                out.append({"target": target, "side": "self" if target == -(slot + 1) else "ally",
                            "species": mon.species if mon else None})
        return out

    def move_entry(self, request_move: dict, slot: int) -> dict:
        """One ``available_moves`` entry (the observation's spelling: id/type/target, plus priority and max_pp)."""
        info = data.move_info(request_move.get("id")) or {}
        dex_target = info.get("target") or request_move.get("target") or "normal"
        targets = target_positions(request_move.get("target") or dex_target, slot)
        return {
            "id": request_move.get("id"),
            "type": str(info.get("type") or "Normal").upper(),
            "category": str(info.get("category") or "Status").upper(),
            "base_power": int(info.get("base_power") or 0),
            "accuracy": _accuracy(info.get("accuracy", True)),
            "priority": int(info.get("priority") or 0),
            "current_pp": int(request_move.get("pp") or 0),
            "max_pp": int(request_move.get("maxpp") or 0),
            "target": _upper_snake(dex_target),
            "targets": targets,
            "target_options": self.target_options(targets, slot),
        }


def target_positions(target_type: str, slot: int) -> list[int]:
    """Legal target integers for a move target type from ``slot`` — fixed by the type, never by who is standing there."""
    me, ally = -(slot + 1), -(2 - slot)
    if target_type == "adjacentAlly":
        return [ally]
    if target_type == "adjacentAllyOrSelf":
        return [ally, me]
    if target_type == "adjacentFoe":
        return [1, 2]
    if target_type in ("normal", "any"):
        return [ally, 1, 2]
    return [0]  # self, field, side and spread moves: "no target", spelled 0


def player_view(snap: dict, side: str) -> BattleView:
    """Rebuild one seat's client view from a bridge snapshot: Team Preview request, then the log, then the current request."""
    other = "p2" if side == "p1" else "p1"
    mine = snap["sides"][side]
    base = {r["species"]: r.get("base_species") or r.get("name") for r in snap["sides"][other].get("roster") or []}
    view = BattleView(side, base_species=base)
    if mine.get("preview_request"):
        view.apply_request(mine["preview_request"], current=False)
    view.feed_all((snap.get("logs") or {}).get(side) or [])
    request = mine.get("request")
    if request and not request.get("wait"):
        view.apply_request(request, current=True)
    return view


# -- draft -----------------------------------------------------------------------------------------------


def draft_state(session_id: str, *, me: str, pool: list[dict], rosters: dict[str, list[dict]], picks: list[dict],
                first_drafter: str, current_seat: str, version: int, them: str | None = None,
                now: datetime | None = None) -> GameState:
    """A draft state. ``rosters`` is ordered seat 0 then seat 1 (seat ids are the agents' ids, as live).
    ``available_cards`` lists every unused card; only ``legal_actions`` applies the Item Clause."""
    taken_items = {data.to_id(c.get("item")) for c in rosters.get(current_seat, [])}
    deadline = (now or datetime.now(timezone.utc)) + timedelta(seconds=DRAFT_TIMEOUT_SECONDS)
    obs = {
        "phase": "draft",
        "draft_complete": False,
        "battle_format": BATTLE_FORMAT,
        "current_seat": current_seat,
        "pick_number": len(picks) + 1,
        "picks_remaining": DRAFT_PICKS - len(picks),
        "first_drafter": first_drafter,
        "available_card_ids": [c["card_id"] for c in pool],
        "available_cards": copy.deepcopy(pool),
        "rosters": {seat: [{"card_id": c["card_id"], "species": c["species"]} for c in cards] for seat, cards in rosters.items()},
        "picks": copy.deepcopy(picks),
        "unused_card_ids": [c["card_id"] for c in pool],
        "decision_timeout_seconds": DRAFT_TIMEOUT_SECONDS,
        "decision_deadline_at": deadline.isoformat(),
        "battle_starting": False,
    }
    my_turn = current_seat == me
    legal = [_action(session_id, f"draft_pick:{c['card_id']}", f"Draft {c['card_id']}", {"type": "draft_pick", "card_id": c["card_id"]}, version)
             for c in pool if data.to_id(c.get("item")) not in taken_items] if my_turn else []
    seats = list(rosters)
    position = seats.index(current_seat) if current_seat in seats else 0
    return _new_state(session_id, "draft", obs, legal, version, actor=current_seat, position=position, my_turn=my_turn)


# -- team preview ----------------------------------------------------------------------------------------


def preview_state(session_id: str, snap: dict, side: str, *, version: int = 0, agent_id: str | None = None) -> GameState | None:
    """Team Preview for ``side`` from a bridge snapshot taken before either lineup is chosen."""
    request = snap["sides"][side].get("request") or {}
    if not request.get("teamPreview"):
        return None
    view = player_view(snap, side)
    obs = {
        "phase": "team_preview",
        "battle_format": BATTLE_FORMAT,
        "battle_tag": snap.get("battle_tag") or f"battle-{BATTLE_FORMAT}",
        "waiting_for_action": True,
        "your_roster": [m.to_obs() for m in view.team.values()],
        "opponent_roster": [m.to_obs() for m in view.teampreview_opponent],
        "clock": dict(PREVIEW_CLOCK),
    }
    action = {"type": "select_lineup", "roster": [m.species for m in view.team.values()], "bring_count": 4, "lead_count": 2,
              "instructions": LINEUP_INSTRUCTIONS}
    legal = [_action(session_id, "select_lineup", LINEUP_LABEL, action, version)]
    return _new_state(session_id, "team_preview", obs, legal, version, actor=agent_id or side, position=0 if side == "p1" else 1)


# -- battle ----------------------------------------------------------------------------------------------


def battle_state(session_id: str, snap: dict, side: str, version: int, *, agent_id: str | None = None) -> GameState | None:
    """Our doubles-turn state from a bridge snapshot, or None when this side has no decision to make."""
    request = snap["sides"][side].get("request")
    if not request or request.get("wait") or request.get("teamPreview"):
        return None
    view = player_view(snap, side)
    actives = view.active_pokemon
    active_obs = [m.to_obs() if m else None for m in actives]
    moves = [[view.move_entry(m, slot) for m in view.available_moves[slot]] for slot in range(2)]
    obs = {
        "battle_tag": snap.get("battle_tag") or f"battle-{BATTLE_FORMAT}",
        "turn": view.turn,
        "format": BATTLE_FORMAT,
        "finished": view.finished,
        "won": view.won,
        "lost": view.lost,
        "weather": dict(view.weather),
        "fields": dict(view.fields),
        "side_conditions": dict(view.side_conditions),
        "opponent_side_conditions": dict(view.opponent_side_conditions),
        "team": {key: mon.to_obs() for key, mon in view.team.items()},
        "opponent_team": {key: mon.to_obs() for key, mon in view.opponent_team.items()},
        "is_doubles": True,
        "force_switch": list(view.force_switch),
        "can_tera": list(view.can_tera),
        "active_pokemon": active_obs,
        "opponent_active_pokemon": [m.to_obs() if m else None for m in view.opponent_active_pokemon],
        "available_moves": moves,
        "available_switches": [[m.to_obs() for m in slot] for slot in view.available_switches],
        "target_legend": dict(TARGET_LEGEND),
        "protocol_log": list((snap.get("logs") or {}).get(side) or []),
        "waiting_for_action": True,
        "clock": dict(BATTLE_CLOCK),
    }
    force = view.force_switch
    reserves = {m.species for slot in view.available_switches for m in slot}
    short_handed = sum(force) > len(reserves)  # both fainted, one reserve: Showdown takes one switch and one pass
    slots = []
    for slot in range(2):
        switches = [{"type": "switch", "species": m.species} for m in view.available_switches[slot]]
        if force[slot]:
            options = switches + ([{"type": "pass"}] if short_handed or not switches else [])
        elif any(force) or actives[slot] is None:
            options = [{"type": "pass"}]
        else:
            options = [_template_move(entry) for entry in moves[slot]] + switches
        slots.append({"slot": slot, "board_position": -(slot + 1), "active": copy.deepcopy(active_obs[slot]),
                      "force_switch": bool(force[slot]), "options": options or [{"type": "pass"}]})
    action = {"type": "doubles_turn", "slots": slots, "target_legend": dict(TARGET_LEGEND), "instructions": DOUBLES_INSTRUCTIONS}
    legal = [_action(session_id, "doubles_turn", DOUBLES_LABEL, action, version)]
    return _new_state(session_id, "moving", obs, legal, version, actor=agent_id or side, position=0 if side == "p1" else 1)


def _template_move(entry: dict) -> dict:
    """A legal-actions move option (the template's spelling: move_id/move_type, no priority/max_pp/target)."""
    return {
        "type": "move",
        "move_id": entry["id"],
        "base_power": entry["base_power"],
        "category": entry["category"],
        "move_type": entry["type"],
        "current_pp": entry["current_pp"],
        "accuracy": entry["accuracy"],
        "targets": list(entry["targets"]),
        "target_options": [dict(t) for t in entry["target_options"]],
    }


# -- agent answer -> Showdown choice ---------------------------------------------------------------------


def to_choice(payload: dict) -> str:
    """{"type":"doubles_turn","slot_0":{...},"slot_1":{...}} -> 'move fakeout 2, switch amoonguss'."""
    parts = []
    for key in ("slot_0", "slot_1"):
        s = payload.get(key) or {"type": "pass"}
        if s.get("type") == "move":
            target = s.get("target")
            parts.append(f"move {s['move_id']}" + (f" {target}" if target not in (None, 0) else ""))
        elif s.get("type") == "switch":
            parts.append(f"switch {s['species']}")
        else:
            parts.append("pass")
    return ", ".join(parts)


def team_text(cards: list[dict]) -> str:
    """Showdown importable team text for a list of cards (in order)."""
    blocks = []
    for c in cards:
        evs = " / ".join(f"{v} {k.upper() if k != 'spe' else 'Spe'}" for k, v in (c.get("evs") or {}).items())
        evs = evs.replace("ATK", "Atk").replace("DEF", "Def").replace("SPA", "SpA").replace("SPD", "SpD").replace("HP", "HP")
        lines = [f"{c['species']} @ {c['item']}", f"Ability: {c['ability']}", "Level: 50"]
        if evs:
            lines.append(f"EVs: {evs}")
        lines.append(f"{c['nature']} Nature")
        lines += [f"- {m}" for m in c["moves"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def preview_choice(bring: list[str], leads: list[str], roster: list[Any]) -> tuple[str, list[int]]:
    """A ``select_lineup`` answer -> Showdown's Team Preview choice over the full six, leads first:
    (``"team 3142"``, 0-based roster indices in battle order). ``roster`` is species ids or cards."""
    ids = [data.to_id(r if isinstance(r, str) else r.get("species")) for r in roster]
    lead_ids = [data.to_id(x) for x in leads]
    order = lead_ids + [data.to_id(b) for b in bring if data.to_id(b) not in lead_ids]
    indices = [ids.index(s) for s in order]
    return "team " + "".join(str(i + 1) for i in indices), indices


def legal_payload_from_choice(payload: Any) -> dict:
    return payload if isinstance(payload, dict) else {}
