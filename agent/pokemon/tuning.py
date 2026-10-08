"""Tunable weights for the code brain, so self-play can measure them instead of us guessing.

Defaults were tuned by self-play on Oct 6 (sim/sweep.py): the combination of the four annotated
changes beat the previous defaults 55.7% ± 3.1 over 1,000 games.

Every number here is a default. A ``PokemonAgent`` carries its own ``params`` dict (so two
versions can play each other in one process with different weights), and the draft, lineup and
battle modules read from whatever dict they are handed. ``agent/versions/tuned.py`` builds an
agent from the ``AGENT_TUNE`` environment variable (a JSON object of overrides) and
``sim/sweep.py`` runs variants against the baseline.
"""

from __future__ import annotations

import json
import os

DEFAULTS: dict[str, float] = {
    # draft
    "draft_offense_w": 10.0,       # x average hit quality into their drafted cards
    "draft_defense_w": 5.0,        # x resist/weak score vs their drafted attacks
    "draft_support_cap": 15.0,     # max support-move bonus (was 30; self-play 1,000 games: +3.9 pts)
    "draft_fast_bonus": 5.0,       # speed >= 120 after EVs/nature
    "draft_veryfast_bonus": 8.0,   # speed >= 150
    "draft_coverage_w": 3.0,       # per new offensive type (max 3)
    "draft_shared_weakness_w": 3.0,
    "draft_denial_margin": 3.0,    # consider denial when our top two are within this
    "draft_denial_gap": 5.0,       # ... and the card is worth this much more to them
    "draft_mode_w": 0.0,           # item 2: value cards that fit the mode we have started (Trick Room / Tailwind / rain / sun) (0 = off)
    "draft_role_w": 0.0,           # item 2: by pick 4+, value missing roles (speed control, Fake Out/redirect, a resist to their best attacker) (0 = off)
    "draft_deny_mode_w": 0.0,      # item 2: deny the card that completes THEIR mode (0 = off)
    # lineup
    "lineup_offense_w": 6.0,
    "lineup_resist_w": 1.5,
    "lineup_speed_w": 1.5,
    "lineup_speed_control": 4.0,
    "lineup_fake_out": 3.0,
    "lineup_quad_penalty": 1.5,
    "lead_se_penalty": 0.8,        # per 2x known move into a lead
    "lead_quad_penalty": 3.0,      # per 4x known move into a lead
    "lead_spread_penalty": 3.0,    # a known spread move that is 2x into both leads
    "lineup_predict_opp": 0.0,
    "lineup_answer_w": 0.0,        # item 1: penalty per top-2 opposing attacker we bring no resist/immunity for (0 = off)
    "lead_priority_multihit_w": 0.0,  # item 1: penalty per lead that a likely opposing lead can KO with priority or multi-hit (0 = off)     # 1 = score against the opponent's PREDICTED four (and leads), weighted by this
    "lead_ohko_w": 0.0,            # penalty per lead a faster opposing set OHKOs at full HP (real damage model, Intimidate applied); 0 = off
    # battle
    "ko_bonus_guaranteed": 25.0,
    "ko_bonus_possible": 10.0,
    "priority_w": 3.0,
    "survival_possible": 0.5,      # value multiplier when a faster possible-KO threatens the slot
    "ally_damage_w": 0.75,         # charge for spread damage into our own ally (was 1.5)
    "protect_bonus_guaranteed": 45.0,
    "protect_bonus_possible": 30.0,
    "switch_threatened_bonus": 15.0,
    "focus_fire_bonus": 30.0,
    "fakeout_base": 90.0,          # (was 60)
    "fakeout_threat_bonus": 20.0,
    "fakeout_setup_value": 50.0,
    "switch_weak_threshold": 20.0,
    "switch_weak_bonus": 15.0,     # (was 10)
    "choice_lock_rerank": 1.0,     # steer a Choice holder away from locking into a move that is dead against a remaining opponent (0 = off)
    "focus_risk": 0.35,            # chance the opponent double-targets a slot that both of its actives can KO together (Oct 7 losses)
    "opp_protect_risk": 0.3,       # chance an opposing Pokémon in KO range Protects (if it can and didn't last turn): discounts focus fire
}


def merged(overrides: dict | None = None) -> dict[str, float]:
    params = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in DEFAULTS:
            raise KeyError(f"unknown tuning parameter {key!r}; known: {sorted(DEFAULTS)}")
        params[key] = float(value)
    return params


def from_env(var: str = "AGENT_TUNE") -> dict[str, float]:
    raw = os.environ.get(var, "").strip()
    return merged(json.loads(raw) if raw else None)


def label(params: dict[str, float]) -> str:
    """Short name listing only what differs from the defaults."""
    diff = {k: v for k, v in params.items() if DEFAULTS.get(k) != v}
    return "tuned:" + ",".join(f"{k}={v:g}" for k, v in sorted(diff.items())) if diff else "tuned:defaults"
