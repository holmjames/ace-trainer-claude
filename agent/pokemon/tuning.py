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
    "draft_v2": 1.0,               # 1 = counter-aware drafter (matchup model + lookahead over their next picks); 0 = the greedy scorer above
    "draft_lookahead": 3.0,        # v2: how many of the opponent's best replies to try at their next pick (0 = greedy rollout only)
    "draft_minimax": 0.5,          # v2: weight on their most damaging reply vs the average of those tried (1 = pure worst case)
    "draft_card_w": 1.0,           # v2: weight on the fitted per-card strengths (0 = matchups, roles and tier adjustments only). Oct 8: 0 + tier_adjust 1.25 lifted Incineroar to top-5 at pick 1 on the Oct 7 pools but showed no gain over the eight live drafts; kept at 1
    "draft_tier_adjust": 0.0,      # v2 + Team Preview v2: weight on data/card_tiers.json's written-down adjustments (0 = off; see draft_card_w)
    "draft_reply_depth": 1.0,      # v2: their consecutive picks to branch on (2 = both picks of a snake double; else greedy)
    "draft_counter_replies": 3.0,  # v2: their best replies by the counter-picker's scorer (agent/pokemon/counterpick.py) tried next to the team model's draft_lookahead best; 0 = team model only
    "draft_counter_prior": 0.02,   # v2: our tier rating (data/card_tiers.json, 1/100 log-odds) x this, added to the counter-picker's score as its strength prior when it picks their replies (0.02: adds the L1 reply Gholdengo to its top three; 0.03+ pushed Incineroar below 5th at L4 pick 1, Oct 8)
    "draft_four_eval": 1.0,        # v2: value a finished draft by the four we would bring (Team Preview's criterion, preview weights) instead of all six (Oct 8: early picks stayed home in four live games); 0 = six against six
    "draft_their_fours": 3.0,      # v2: how many of their likeliest fours that value is taken against (the mean / worst blend uses lineup_v2_robust)
    "draft_tiebreak_eps": 0.01,    # picks whose finished-draft win chance is within this of the best are a tie, broken by the match's secret seed (0 = always the best)
    "lineup_tiebreak_eps": 0.01,   # same for the four we bring and the lead pair (win chance of the candidate value / the lead value)
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
    "lineup_v2": 1.0,              # 1 = Team Preview from the matchup model (their likely four and leads); 0 = the scorer above
    "lineup_v2_robust": 0.25,      # v2: weight on our four's worst case against any four they could bring (rest: their likely mix)
    "lineup_v2_template": 0.5,     # v2: share of their predicted leads taken from the "Fake Out / speed control + best spread attacker" template
    "lineup_v2_lead_robust": 0.0,  # v2: weight on our lead pair's worst opening against any plausible lead of theirs (vs the expected one)
    "lineup_v2_four_template": 0.0,  # v2: share of their predicted FOUR from the same template (off: on the Oct 7 opponent's live lineups M1 38->34%, M4 24->0%, M2 unchanged)
    "lead_veto_w": 0.5,            # v2 lead veto: a pair that loses a lead on turn 1 (or leads an exposed speed-control setter) against their two likeliest openings is never chosen while a pair of the same four is safe; with no safe pair, this much per lead lost (half for an exposed setter). 0 = off
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
    "hp_units": 1.0,               # value damage in % of the target's MAX HP (capped at what it has left), not % of current HP (+3.7 pts, 2,000 games, Oct 8)
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
