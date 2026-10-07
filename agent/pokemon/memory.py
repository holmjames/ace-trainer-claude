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
                card = self.pool_cards.get(entry.get("card_id") or "", entry)
                merged = {**entry, **card} if card is not entry else entry
                key_species = species_key(merged.get("species"))
                if key_species:
                    target[key_species] = merged

    # -- team preview ----------------------------------------------------------------

    def observe_team_preview(self, obs: dict) -> None:
        for entry in obs.get("your_roster") or []:
            self._enrich(self.my_cards, entry)
        for entry in obs.get("opponent_roster") or []:
            self._enrich(self.opp_cards, entry)

    def _enrich(self, side: dict[str, dict], entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        key = species_key(entry.get("species") or entry.get("name"))
        if not key:
            return
        side[key] = {**side.get(key, {}), **{k: v for k, v in entry.items() if v is not None}}

    # -- battle ----------------------------------------------------------------------

    def observe_battle(self, obs: dict) -> None:
        opponent_team = obs.get("opponent_team")
        if isinstance(opponent_team, dict):
            for summary in opponent_team.values():
                if isinstance(summary, dict):
                    key = species_key(summary.get("species"))
                    if key and key not in self.opp_lineup_seen:
                        self.opp_lineup_seen.append(key)

    def record_turn(self, **fields: Any) -> None:
        self.turns.append(fields)

    # -- prompt material -------------------------------------------------------------

    def known_sets(self, side: str) -> list[dict]:
        """Compact card summaries for the prompt: what we know for sure."""
        cards = self.my_cards if side == "mine" else self.opp_cards
        out = []
        for card in cards.values():
            out.append(
                {
                    "species": card.get("species"),
                    "item": card.get("item"),
                    "ability": card.get("ability"),
                    "nature": card.get("nature"),
                    "types": card.get("types"),
                    "base_stats": card.get("base_stats"),
                    "evs": card.get("evs"),
                    "moves": card.get("moves"),
                }
            )
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
