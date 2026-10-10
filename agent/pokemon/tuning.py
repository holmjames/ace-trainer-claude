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
    "draft_first_pick_w": 1.0,     # B1: weight on the first-pick policy (draft.first_pick_bonus) at our first pick, half at our second, 0 after; 0 = search only
    "draft_first_strength_w": 1.0,   # B1: x the candidate's mean duel margin over the remaining pool
    "draft_first_scarcity_w": 0.5,   # B1: x role scarcity (1 / holders in the pool, summed over Fake Out, Intimidate, speed control, redirection, Ghost); 0.5 lifts Incineroar to 5th / 4th at pick 1 on L2 / L3 (0.3: 5th / 7th)
    "draft_first_exposure_w": 0.2,   # B1: x (pool cards that beat the candidate by >= 0.2) / 6: the counters the opponent can still take
    "draft_first_guard_w": 0.1,      # B1: x the hand guard list for the live catalog (+1 / -1 / 0)
    "draft_first_adjust_w": 1.0,     # B1: x the per-card first-pick table (draft.first_pick_adjust: data/first_pick_adjust.json, all 0 = no change; DRAFT_FIRST_PICK_ADJUST=<path> swaps the table for a simulator A/B arm)
    "draft_hole_w": 0.5,           # B4: log-odds per coverage hole in the finished-draft value (their card none of our four hits for 40% of its HP with its best real attack; immunities and Air Balloon count); 0 = off
    "draft_sleep_w": 0.0,          # their sleepers against our sleep-proof cards in the finished-draft value (one-sided: x matchup.Board.sleep_reach of their four into ours; a Spore user against a four with no sleep-proof card is 1, one 0.5, two 0). 0 = off, the deployed draft: unmeasured at 0.4 it changed 21 of 120 live picks (Oct 9 review); set it from a simulator series
    "draft_tiebreak_eps": 0.01,    # picks whose finished-draft win chance is within this of the best are a tie, broken by the match's secret seed (0 = always the best)
    "draft_tiebreak_exposure": 0.0,  # > 0: inside that window only the cards the fewest undrafted pool cards beat (ScoredCard.exposure: the counters the opponent can still take) go to the seed's draw (Oct 9 draft review: the lower-exposure card won 61 % of 171 near-tie pairs). 0 = the draw over the whole window, bit for bit; off until its simulator A/B
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
    "lineup_hole_w": 0.5,          # B4: per coverage hole, in Team Preview v2's pairing (log-odds) and the old scorer's score_four (points); 0 = off
    "lineup_sleep_w": 0.0,         # the same both ways in Team Preview v2's pairing (matchup.Board.sleep_pressure). 0 = off, the deployed preview: at 0.4 their Amoonguss fours in 51c86e5c went 0 -> 46%, but under the live seeded tie-break no Rillaboom four was brought there any more (45/64 seeds -> 0) and 3 of 20 live previews changed (Oct 9 review); set it from a simulator series
    "lead_veto_w": 0.5,            # v2 lead veto: a pair that loses a lead on turn 1 (or leads an exposed speed-control setter) against their two likeliest openings is never chosen while a pair of the same four is safe; with no safe pair, this much per lead lost (half for an exposed setter). 0 = off
    # the deep Team Preview (agent/pokemon/preview_sim.py): simulated openings for every plan (four + leads) under a deadline
    "lineup_sim_budget_s": 20.0,   # seconds of the 85 s Team Preview clock the search may spend (the model call gets what is left); 0 = off, the stage2 preview exactly
    "lineup_sim_turns": 4.0,       # turns per simulated opening
    "lineup_sim_fours": 10.0,      # at most this many of our fours (the four-against-four game's best), every lead pair of each
    "lineup_sim_four_margin": 0.15,  # ... and only fours within this win chance of the best four
    "lineup_sim_their_fours": 8.0,  # their likeliest fours the plans are played against (plus the search-bot template's four): their actual four was among 8 in 17 of the 19 live previews, among 6 in 15
    "lineup_sim_expected_w": 0.7,  # a plan's value: this x its expected margin over their fours + the rest x its worst four
    "lineup_sim_four_w": 0.5,      # ... + this x its four's v2 value (log-odds, relative to v2's best four): the bench, which a few simulated turns barely see
    "lineup_sim_keep_v2": 0.05,    # v2's own plan (its four and lead veto pair) stays code #1 unless the search's best beats it by more than this (Pokemon); it plays the last round too
    "lineup_sim_beta": 2.0,        # how sharply their four distribution follows their value against our likely fours (softmax, per log-odds)
    "lineup_sim_flat": 0.5,        # share of their four distribution spread evenly over the fours simulated (live: the predicted top four was right in 4 of 19)
    "lineup_sim_lead_floor": 0.2,  # share of each four's lead distribution spread evenly over its six pairs (live: their actual lead pair got 0.26 on average, 0 twice)
    "lineup_min_model_s": 10.0,    # after the search, a Team Preview model call needs at least this many seconds of the clock, else code #1 is played
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
    "protect_prior_rows": 0.0,     # 1 = every row's damage into a foe with a fresh Protect is discounted by the search tally's Protect rate for its threat (battle.rank_candidates.protect_risk), not only focus fire's KOs by the flat opp_protect_risk. Oct 8 live set: m2_t3 and m1_t3 pass, no position lost; off until a series (test_tuning.py:44's +100 KO-bonus check fails with it on)
    "protect_aimed_risk": 0.62,    # with protect_prior_rows: the Protect rate of a foe facing our KO that one of our single-target moves aimed at last turn (4/6 pooled)
    "ko_threat_w": 0.5,            # KO bonus scales by the threat removed (its best action into us, in damage points) x this (Oct 8)
    "exposure_w": 1.0,             # charge for the EXTRA KO risk a move creates for its user (stat drops, recoil, Glaive Rush) x its value
    "reply_mixture": 1.0,          # 1 = rank candidates by the reply mixture (agent/pokemon/replies.py); 0 = the scorer above. On since Oct 9 (paired simulator A/B vs the search bot's newest version at its 15 s preview)
    "reply_seed": 0.0,             # the mixture's tie-break seed: 0 = a secret per-match seed (logged), else this value (tests, replays)
    "reply_worst_w": 0.15,         # the mixture's risk aversion: the row's worst plausible reply weighted into its value
    "reply_floor_margin": 100.0,   # never a row whose worst plausible reply trails the best worst case by more than this (one Pokémon)
    "reply_loss_w": 1.0,           # the mixture's charge for one of ours KO'd (x its material) and reply_chip_w for damage taken
    "reply_chip_w": 0.35,
    "reply_next_w": 0.4,           # the mixture's one-ply next-turn term: next turn's KOs each side can take, discounted by this
    "reply_protect_tempo": 8.0,    # the mixture's charge per Protect of ours: the turn it gives away
    "reply_search_qr": 0.85,       # weight of the search class's own read of the turn (replies._search_reply: a quantal response to our likely rows) in its prior, the rest the tally prior; 0.85 scores the bot's actual replies best (32 replayed turns: log-likelihood -1.31 vs -1.35 at 1.0, -2.46 for the tally) and keeps every read below 0.9
    "reply_qr_beta": 0.06,         # that read's sharpness over our rows (twice this over its replies), in damage points
    "reply_qr_rounds": 4.0,        # rounds of that read, from the class prior
    "sleep_future_w": 0.5,         # Part 3: weight of a later turn's action against this turn's (the turns a sleep move takes, a Taunt blocks)
    "sleep_use_prior": 0.6,        # Part 3: chance a sleep user with a target uses its sleep move this turn (aimed 0.8 at the slot that loses most)
    # model layer (ops lane)
    "override_margin": 0.15,       # E2: a free-form override whose value trails the best row by more than this share of the top value is rejected, unless the model's opponent prediction matches the code's top column
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
