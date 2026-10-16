# Ace Trainer

An AI agent that drafts and battles Pokémon (VGC doubles) for the AltruAgent AI Agent
Gaming Tournament at UCLA. Python does the math; Claude Opus judges.

**Case study:** [jholm.co/ace-trainer](https://jholm.co/ace-trainer/), one real turn
explained step by step.

- **Draft (code):** scores every card in the pool against what the opponent can still
  take. A pick takes about 90 ms.
- **Team Preview (code + Opus):** simulates the opening turns of each lineup for up to
  20 seconds; Opus makes the final call.
- **Every battle turn:** Python builds a damage sheet (every hit both ways, speed order,
  threats) and ranks complete plays in about 10 ms. Claude Opus 5.5 picks one in about
  5 seconds, and the code can overrule a pick that scores far below the best option.
- **Markov by design:** every decision depends only on the current state of the game,
  battle log included. Restarted from scratch mid-battle in 1,200 simulated games, it
  finished with the same result in 1,188.
- **Tested before shipping:** 83,000 simulated games; every change was compared with the
  previous build on identical drafts before it went live.

The full description the tournament rules ask for is in
[This entry: the Pokémon agent](#this-entry-the-pokémon-agent). The rest of this README
is the upstream AltruAgent starter's documentation of the runtime, kept as is apart from
notes for this fork.

## Built on the AltruAgent starter

A Python starter kit for building your agent for the AltruAgent tournament.
This is the repository you build your agent in. The platform itself
(`Agent_ACP`) is a separate, read-only reference you don't need to touch or
run locally.

**One way to run your agent:** put your **Official Agent Key** in `.env` and
run

```bash
python -m agent --tournament           # your tournament games
python -m agent --match                # your test matches (Testing page)
python -m agent --tournament --match   # both, in one process
```

Leave it running. It picks up those games by itself and plays each one with
your `create_agent()`. Nobody copies an id and nobody claims anything. If
several games are assigned at once, it plays all of them **at the same time**,
each in its own process with its own fresh agent instance. While it waits it
uses **no AI tokens**: it only asks the platform every ~10 seconds whether a
game is ready. Only playing a game with an LLM agent uses tokens.

Gameplay runs through the platform's generic MCP contract
(`get_game_state`/`wait_for_update`/`play_action`/...), so the runtime is
the same for Pokémon, Werewolf and Red Alert (see [`GAMES.md`](GAMES.md)):
only your `choose_action` needs to know how each game's moves look.

Step-by-step guide on the tournament site:
<https://platform.altruagent-game.com/tournament/agent-guide>

## This entry: the Pokémon agent

Entry for the AltruAgent AI Agent Gaming Tournament, Pokémon Showdown only
(`pokemon_vgc_doubles_draft`: snake-draft 6 of 18 shared cards, bring 4, play a
4v4 doubles battle). Werewolf and Red Alert are not entered. Submitted by
James Holm (`holmjames`). This section is the disclosure the Official Rules
(§8) ask for: how the agent runs, what it is made of, and what is not
published.

**Submitted version:** the latest commit on this repository's default branch
at the Submission Deadline; its full commit ID is entered on the tournament
dashboard. The agent's code is that of commit `b74574a`; the documentation
commits on top of it change only `README.md` and `PLAN.md`. Every match log
starts with a `config` line naming the commit that played it (see *Records*
below).

### How it runs

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"             # installs the pinned anthropic SDK (1.12.0) and httpx (0.28.1) with the rest
cp .env.example .env                # then fill in the secrets below
python -m agent --check-tournament  # key, connection, agent factory, pinned versions, one tiny model call: all ✓
./scripts/run_tournament.sh         # tournament day: python -m agent --tournament, detached, auto-restart, logged
```

`.env` holds (never committed): `ALTRUAGENT_OFFICIAL_AGENT_KEY` (the platform
credential), `ANTHROPIC_API_KEY` and, for a key that spans workspaces,
`ANTHROPIC_WORKSPACE_ID`. Without `ANTHROPIC_API_KEY` the same agent plays in
code-only mode (no model calls), and `--check-tournament` fails on purpose so
that this cannot go unnoticed. Optional overrides: `AGENT_MODEL`,
`AGENT_FALLBACK_MODEL`, `AGENT_EFFORT`, `AGENT_LLM_TIMEOUT`,
`AGENT_FALLBACK_TIMEOUT` (defaults in `agent/llm/anthropic_provider.py`),
`AGENT_LOG_DIR`, `AGENT_VERSION` (a label for the logs) and
`AGENT_CAPTURE_DIR` (where raw server states are saved). The competition
configuration uses the code defaults; the `config` line of each match log
records the models, timeouts, effort and tuning that actually played.

`scripts/run_tournament.sh` runs that pre-flight check first, then
`python -m agent --tournament` (under `caffeinate -dims` on a Mac) in a restart
loop detached from the terminal, writing everything to
`logs/runtime-tournament-<timestamp>.log`; `--status` and `--stop` manage it.
It sets `AGENT_CAPTURE_DIR=tests/fixtures/live`, so the raw server states of
each match are saved (see *Records*).

### What makes the decisions

| Decision | Who decides | Where |
|---|---|---|
| Draft pick (15 s clock) | Code only, no model call: a search that plays the rest of the snake draft out for every offered card against the opponent's likely replies. | `agent/pokemon/draft.py`, `matchup.py`, `counterpick.py` |
| Team Preview (bring 4, lead 2) | Code ranks fours and lead pairs (a four-against-four game, then about 20 s of simulated openings) and sends its top three with a matchup table; Claude chooses the lineup; the starter's validator checks it. | `agent/pokemon/lineup.py`, `preview_sim.py`, `agent/agent.py` |
| Each battle turn | Code builds a "turn sheet" (speed order, damage estimates, threats, warnings, switch-in checks) ending in up to six complete candidate turns ("rows"); Claude picks a row or proposes another joint action; a code gate can overrule it; the validator checks the result. | `agent/pokemon/battle.py`, `opponent.py`, `agent/agent.py` |

### Model services

- **Anthropic Messages API** through the official `anthropic` Python SDK
  (`agent/llm/anthropic_provider.py`). The chain is `claude-opus-5-5`, then
  `claude-sonnet-5-5` when Opus errors or times out, then the computed move.
  Per-call timeout ceilings are 18 s (Opus) and 8 s (Sonnet), each with a 2 s
  wall-clock guard on top, so a hung connection cannot hold a call longer.
  Inside a decision every call gets only what the clock budget below leaves:
  Opus may use the larger of (budget − 6 s) and 70 % of the budget, up to its
  ceiling; Sonnet gets what is left, up to its own ceiling; a model with under
  3 s left is not called. Effort (`output_config.effort`) is `low` for battle
  turns and `high` for Team Preview. Answers are strict JSON
  (`output_config.format` with a JSON schema), `max_tokens` is 4,000, and no
  `thinking` parameter is sent. Both system blocks (the fixed prompt and this
  match's known sets) are marked cacheable. The SDK's own retries are off; the
  agent owns retries and fallback.
- **What the model receives:** only this seat's game state as the server
  delivered it (the trimmed observation and the per-slot options with every
  target named), the drafted sets that the draft made public to both players,
  what this match's battle log has shown, and the code's computed notes. The
  fixed system prompt adds general rules and per-class opponent tendencies
  (see *Fixed reference materials*). Nothing else: no spectator data, no other
  matches, no internet lookups. The agent makes no network calls other than the
  platform's MCP runtime (upstream code in `altruagent/`) and the Anthropic API.
- **Prompts:** the system prompt, its two switched-on variants and the Team
  Preview instructions are in `agent/pokemon/prompts.py`; the per-decision
  user message is the JSON payload built in `agent/agent.py` (`_lineup`,
  `_turn`).
- **Public reasoning:** the model's `reasoning_summary` is sent with the move
  as its public one-line reasoning (cleaned, at most 280 characters). When the
  code does not play what the model proposed (an override refused, a row sent
  back to row 0), the played row's own `why` is sent instead, so the public line
  describes the move that was made. Without a model answer the top row's `why`
  is sent; a draft pick sends "Drafted <species>.".

### The clock

With every Team Preview and battle request the server reports the time left:
about 50 s for a battle decision, 85 s at Team Preview, and a 420 s bank per
side for the whole battle.
When a decision runs out the platform plays a move for us, and an empty bank
forfeits. `_clock_budget` in `agent/agent.py` turns the server's numbers into
one model deadline per decision:

- **Battle turn:** the smaller of (decision time left − 8 s) and
  (bank − 90 s) ÷ the turns the position still needs, capped at 20 s. The turn
  estimate is 1.75 × the Pokémon left on both sides, at least 4 (a fresh 4v4:
  14). A full bank therefore gives 20 s a turn, and a long game slows down
  before the bank can run out.
- **Team Preview:** the code's search (about 20 s) comes off the decision first;
  the model gets the smaller of (what is left − 8 s) and a fifth of the bank
  above 90 s, capped at 60 s. With under 10 s left after the search, code #1
  plays without a model call.
- A deadline under 6 s skips the model and plays the computed move at once.
  A second call (the retry after an invalid answer, or the lethal recheck) is
  made only while at least 6 s of the deadline remain.
- After two decisions in a row on which every model failed, the next decision
  skips the model (a circuit breaker), and the one after it tries again.

### Validation, the code gate, and fallbacks

- **Validation.** Every answer is checked against the server's legal options by
  the starter's validators (`examples/llm/pokemon.py`). An invalid answer is
  retried once with the error quoted, time permitting. If it is invalid again,
  or every model fails, the computed move plays: code #1 at Team Preview, the
  top row in battle.
- **The row gate (battle turns).** A row other than row 0 plays only when its
  score is within 15 % (`override_margin`) of row 0's, measured as
  0.15 × max(row 0's score, 1); otherwise row 0 plays. A proposed joint action
  that is not a row (an "override") is validated, scored with the same scorer
  as the rows, and plays only when it is within the same margin of row 0 or the
  model's `opponent_prediction` matches the code's own top prediction for both
  opposing slots. Two overrides are always refused, prediction or not: a
  Protect from our last Pokémon when it has nothing to wait for (it only
  delays), and a Protect that would lock an unlocked Choice Scarf, Band or Specs
  holder when no listed row Protects that slot (`choice_protect_guard`). A
  refused override plays the model's chosen row, under the same margin. At Team
  Preview there is no value gate: any legal lineup the model names plays, and
  the instructions ask it to justify leaving code #1 against the table.
- **The lethal recheck** (`recheck_v2`): when a valid battle answer attacks with
  a slot that a LETHAL warning or an EXPOSED note names, while row 0 Protects or
  switches that slot and nothing in the answer removes the threat first (our
  hit, or both our hits together, landing first with at least an even KO
  chance, or a Fake Out that flinches it), the model is asked once more with
  the warning and row 0 quoted. Its second answer stands, through the same
  gate; if that call fails or is invalid, the first answer stands. It never
  fires for our last Pokémon with nothing to wait for.
- **Exceptions.** A crash anywhere in the decision code plays the starter's
  always-legal smoke move (or, if even that cannot read the request, a minimal
  legal payload built from the raw request) and is logged. The agent can lose
  a turn to a bug, never a match. Some parts also have their own fallback:
  the draft search falls back to the older greedy scorer, Team Preview v2 to
  the older lineup scorer, the simulated openings to v2's lineup; a failed
  preview table is left out, and a failed prompt variant sends the base
  prompt.

### Fixed reference materials and parameters

All of these are files in this repository, frozen with the code, read-only
while the agent plays, and the same for every opponent.

- `data/pokedex.json`, `data/moves.json`: species and move tables vendored
  from Pokémon Showdown's open-source data by `scripts/build_dex.py`.
- `data/cards.json`: the platform's fixed card sets (48 cards), as offered in
  the draft states of our own test matches before the deadline
  (`scripts/pool_from_live.py`). Read for a card this match never offered us
  (the opponent's first pick when they draft first) and to rebuild memory
  after a restart.
- `data/draft_model.json`: the team model's weights (log-odds per team feature
  and per card), a logistic regression fitted by `scripts/fit_draft_model.py`
  on 16,000 simulator games (random drafts, the code brain playing both
  sides). The draft search and Team Preview score teams with it.
- `data/draft_card_adjust.json`: per-card corrections to the draft search's
  value of a finished draft, a logistic fit on 13,841 of our simulator games
  against a clone of a search-based opponent; applied at half size (see the
  defaults below).
- `data/first_pick_adjust.json`: the first-pick table. Its only non-zero
  entries favour Archaludon (+1.5 log-odds when we draft first, +0.7 when they
  do), in full at our first pick and half at our second.
- `data/card_tiers.json` (written up in `docs/card_tiers.md`): each card's
  rating, used as the strength prior of the counter-picker that proposes the
  opponent's replies in the draft search. Its hand-written adjustments are off
  (`draft_tier_adjust` 0).
- Opponent priors: per opponent *class* (search bot, generic LLM, random; the
  class is read from this match's public draft reasons, pick timing and play),
  how often a Pokémon Protects, switches, sets up or uses Fake Out in a given
  position. They were tallied from earlier test matches and are frozen
  in `agent/pokemon/replies.py` (`SEARCH_BOT_SEEDS`) and in the class lines of
  the system prompt. No opponent is identified by name.
- `agent/pokemon/tuning.py`: every numeric weight and switch; the defaults
  that are on are listed below.
- No model weights, no fine-tunes, and nothing that learns during or between
  matches.

### Defaults that are on (`agent/pokemon/tuning.py`)

A `PokemonAgent` built by `create_agent()` uses `tuning.DEFAULTS` unchanged.
The switches and weights that shape its play:

*Draft*
- `draft_v2` 1: the counter-aware draft search (0 would be the older greedy scorer).
- `draft_lookahead` 3, `draft_counter_replies` 3, `draft_minimax` 0.5,
  `draft_reply_depth` 1: at their next pick the search tries the team model's
  three best replies and the counter-picker's three best, weighting the most
  damaging reply half against the average.
- `draft_four_eval` 1, `draft_their_fours` 3: a finished draft is valued by the
  four we would bring against their three likeliest fours (worst case weighted
  by `lineup_v2_robust` 0.25).
- `draft_card_w` 1: the team model's fitted per-card strengths are included.
- `draft_counter_prior` 0.02: the tier rating as the counter-picker's strength prior.
- `draft_first_pick_w` 1 (strength 1, scarcity 0.5, exposure 0.2, guard 0.1,
  table 1): the first-pick policy, in full at our first pick and half at our second.
- `draft_hole_w` 0.5: a penalty for each opposing card that none of our four
  hits for 40 % of its HP.
- `draft_support_w` 0.25: a counter counts for less when the countered card
  has partners that blunt it (Intimidate, a safe switch-in); both sides.
- `draft_card_adjust_w` 0.5, `draft_card_adjust_them_w` 0.5: the per-card
  table above, never allowed to lower the chance that the pick is Archaludon.
- `draft_tiebreak_eps` 0.01, `lineup_tiebreak_eps` 0.01: among draft picks,
  fours and lead pairs within 1 % win chance of the best, a secret per-match
  seed (drawn with `secrets`, logged) decides, so the public code does not
  predict our choice.

*Team Preview*
- `lineup_v2` 1 with `lineup_v2_robust` 0.25, `lineup_v2_template` 0.5,
  `lineup_hole_w` 0.5, `lead_veto_w` 0.5: fours from the four-against-four
  game, leads from a two-turn opening model, and a lead veto.
- `lineup_sim_budget_s` 20 (`lineup_sim_turns` 4, `lineup_sim_fours` 10,
  `lineup_sim_four_margin` 0.15, `lineup_sim_their_fours` 8,
  `lineup_sim_expected_w` 0.7, `lineup_sim_four_w` 0.5, `lineup_sim_keep_v2`
  0.05, `lineup_sim_beta` 2, `lineup_sim_flat` 0.5, `lineup_sim_lead_floor`
  0.2): the simulated openings; `lineup_min_model_s` 10.

*Battle*
- `hp_units` 1: damage is valued in % of the target's maximum HP, capped at
  what it has left.
- `ko_threat_w` 0.5, `exposure_w` 1, `focus_risk` 0.35, `opp_protect_risk` 0.3,
  `fake_out_use` 0.5, `sleep_future_w` 0.5, `sleep_use_prior` 0.6: a KO is
  worth more the bigger the threat it removes; a move that exposes its user
  (stat drops, recoil) is charged; a slot both foes can KO together is at risk
  35 % of the time; a foe in KO range Protects 30 % of the time; a fresh
  opposing Fake Out is used half the time; sleep moves are priced.
- `choice_lock_rerank` 1: a Choice holder is steered away from locking into a
  move that is dead against a remaining opponent.
- `foe_sleep` 1: a sleeping, paralysed or frozen foe's KO threat counts only
  as often as it gets to act.
- `choice_protect_guard` 1: a Protect that would lock an unlocked Choice item
  is credited nothing in the rows, and refused as an override (see the gate).
- `hp_power_first` 1: Eruption / Water Spout are credited at the HP their
  user likely has when it moves, after the hits of foes that move first; a KO
  counts as guaranteed only if it still lands after those hits.
- `lone_foe_split` 1: when only one of theirs can act this turn, its
  single-target hit can land on one of our Pokémon, not both; its knockout
  threat counts at each of our slots by how likely it is to aim there.
- `speed_end_stall` 1: when Trick Room or their Tailwind ends after this turn
  and a foe out-speeds our slot only because of it, a Protect for that slot is
  credited with the attack it can make first next turn.

*Model layer*
- `gate_repeat_protect` 1: the row gate never sends the model back to a row 0
  that rests on a repeat Protect (a Protect on a streak: 1 in 3, then 1 in 9).
- `override_margin` 0.15: the row gate above.
- `recheck_v2` 1: the lethal recheck as described above.
- `prompt_v2` 1: the system prompt's Protect procedure treats DOUBLE-TARGET as
  a risk, not a forecast, leads its lessons with three from live games, and
  defines `opponent_prediction` in the code's words.
- `prompt_kb` 1: the system prompt carries a battle reference of nine general
  mechanics facts (`prompts.KNOWLEDGE_BASE`).

Present but off by default (tested, not shipped: they cost games against the
other practice opponents): `foe_aim_split` (two foes' threats split by aim),
`mix_play` / `mix_margin` (mixed play among near-equal rows), `dodge_adapt`
(discount attacks into a proven dodger) and `preview_speed_mode` (Team Preview
speed-plan check).

*Off by default:* the reply mixture (`reply_mixture` 0, and `answer_rules_v2`,
which acts only with it), `protect_prior_rows`, `draft_tier_adjust`,
`draft_sleep_w`, `lineup_sleep_w`, `draft_tiebreak_exposure`, `draft_mode_w`,
`draft_role_w`, `draft_deny_mode_w`, `lineup_predict_opp`, `lineup_answer_w`,
`lead_priority_multihit_w`, `lead_ohko_w`, `lineup_v2_lead_robust`,
`lineup_v2_four_template`. Of `agent/pokemon/replies.py` only the opponent
priors are read.

### Memory and isolation

`create_agent()` is called once per match, in that match's own process. All
match memory (`agent/pokemon/memory.py`) lives on that object and dies with
the process; the per-match caches of the draft, matchup and preview modules
are emptied when a match starts (`agent/pokemon/match_state.py`). While the
agent plays, nothing is read from earlier matches, logs or captures, or from
any shared store, and concurrent matches share nothing. Apart from `.env` and
the git commit id written to the log, the only files read are the fixed
reference data above. The only writes are the records below.

**Restart rebuild.** If a match's process is restarted mid-battle, the new
agent has no draft or Team Preview memory. It rebuilds what it can from the
battle itself: every species on either side that the observation or the
battle log shows (including both Team Preview rosters) is filled from
`data/cards.json`, our four are recovered from who has been seen on the field,
fainted or offered as a switch, and everything read from the battle log (who
came in when, Protect streaks, Choice locks, lost items, field effects) is
rebuilt every turn anyway. A card the catalog lacks is played as a placeholder
(the dex's typing and base stats plus the moves shown so far) with a loud
console line. A Terastallized Pokémon's card keeps its species' own typing; the
battle reads the Tera type from the server's summary.

### Records

- `logs/<session_id>.jsonl`: one JSON line per decision (what the model saw,
  what it answered, which model answered, per-call latency and timeout, the
  clock budget, validation errors, how the gate resolved the answer, the
  recheck outcome, the final payload). The first line is a `config` record: git
  commit, agent version label, model chain with timeouts and effort, clock
  settings, tuning parameters, Python and SDK versions. A summary line
  (decisions by Opus, by Sonnet and by code; failures; circuit-breaker skips)
  closes each match on the console and, when our move ends it, in the log.
- `logs/runtime-tournament-<timestamp>.log`: the runtime's own output, from
  `scripts/run_tournament.sh`.
- `tests/fixtures/live/<session_id>/`: the raw server state of each decision,
  saved because the launcher sets `AGENT_CAPTURE_DIR`.
- Anything key-shaped is redacted before a log line is written. `logs/` is
  gitignored and kept locally with a redacted copy of `.env` for at least 30
  days after results, per Official Rules §8–9.

### Not published, and what it does

`.env` (gitignored): the Official Agent Key authenticates this agent to the
platform; the Anthropic API key and workspace id authenticate the model calls.
Neither carries any game information or human input. There are no other
undisclosed components.

### Development-only code (not used in competition)

None of this is imported by `python -m agent`:

- `sim/`: the local simulator. A Pokémon Showdown engine bridge (`bridge.js`,
  `npm install` in `sim/`), `translate.py` (rebuilds the live server's exact
  payloads from the engine's battle log), `harness.py` (N games between any two
  players, seats alternating), `sweep.py` (tuned variants against the
  baseline), `draft_eval.py` (paired games on the same pool and battle seed,
  seats swapped) and `paired_ab.py` (its paired read-out), `clock.py` (the live
  clock in the simulator), `subprocess_player.py` with `shims/checkout_shim.py`
  (an agent from another checkout in a child process), `human.py` (a terminal
  seat so a person can play the agent), `style_stats.py` (play-style statistics,
  such as switch and Protect rates, from a side's battle log) and `cards.json`
  (the simulator's card pool). Players that set `wants_opponent_view` are shown
  the other seat's observations (simulator only), which the reader bot below uses.
- `agent/versions/reader.py`: a practice opponent that predicts the code's top
  play from the other seat's view and best-responds (Protect or switch the slot
  it would hit). It reproduces the live opponents that read our play.
- `agent/versions/`: players the simulator and the arena can seat:
  `code_only` (no model calls), `naive_battle` (the starter's smoke move each
  turn), `random_draft`, `draft_v1` (the older drafter and Team Preview),
  `tuned` (overrides from `AGENT_TUNE`), `model_variant` (another judging
  model), `sparring` (the starter's LLM example on a Claude model),
  `counter_picker` (a search-bot-style counter-picking drafter) and `pivot` (a
  switch-heavy practice opponent: the code brain plus a style layer that
  switches, pivots and spends turns on status). `agent/arena.py` seats two
  versions in one Testing match.
- `scripts/`, except `run_tournament.sh` (the launcher above): data builders
  (`build_dex.py`, `pool_from_live.py`, `catalog_refresh.py`, `build_tiers.py`,
  `fit_draft_model.py`, and the `gen_*_fixture.js` calculator fixtures),
  evaluation and review tools (`scenarios.py`, `live_scenarios.py`,
  `live_positions.py`, `grade_overrides.py`, `draft_eval.py`, `lineup_sweep.py`,
  `preview_replay.py`, `ship_gate.py`, `dry_run.py`, `nightly.sh`,
  `postmortem.py`, `loss_to_scenario.py`, `tally.py`), `run_match.sh`,
  `setup_linux.sh`, and the upstream starter's check and smoke scripts.
- `tests/` (`pytest`, no network, no keys) and its fixtures; `docs/`.

## What I built on top of the starter

The upstream starter (UCLA-Trustworthy-AI-Lab/altruagent-starter) provides the
runtime that talks to the platform (`altruagent/`), the example agents
(`examples/`, including the LLM example and its Pokémon validators), and the
check scripts. Everything that plays Pokémon is this fork's: `agent/pokemon/`,
`agent/llm/`, `agent/versions/`, `agent/arena.py`, `data/`, `sim/`, most of
`scripts/`, and their tests. The placeholder `agent/agent.py` became the agent
described here. All of it is described from the code as it stands; file names
link to where each piece lives.

**No language model was trained.** No weights were changed and no fine-tune was
made. The agent is an existing model (`claude-opus-5-5`, with
`claude-sonnet-5-5` as the fallback) behind a harness of code that does the
arithmetic and limits the model to choosing among computed options. The
numbers the code uses are the hand-set weights in `agent/pokemon/tuning.py`,
adjusted after simulator runs and then frozen, and two small logistic
regressions fitted offline on simulator games (`data/draft_model.json`,
`data/draft_card_adjust.json`). That is parameter fitting for hand-written
scoring rules; nothing learns during or between matches.

### Decision logic (`agent/agent.py`, `agent/pokemon/`)

`PokemonAgent.choose_action` routes each server request by its first legal
action and wraps everything in a guard: an exception anywhere costs one turn
(the starter's always-legal smoke move is played) and never the match.

- **Draft** (`draft.py`, `matchup.py`, `counterpick.py`): code only, because
  the draft clock is 15 seconds; a pick takes well under a second. Every card is
  a complete public set, so `matchup.py` runs a 1-on-1 damage race between
  every two cards in the pool (plain, under either side's Tailwind and under
  Trick Room) on the verified damage model. For each offered card the search
  plays the rest of the snake draft out, trying the opponent's likely replies
  at their next pick: the three best by our team model and the three best by
  a counter-picker that values roles and "beats X" the way a counter-drafting
  opponent does. A finished draft is valued by the team model (`data/draft_model.json`)
  as the four we would bring against their likeliest fours, with the coverage
  hole, support and per-card terms listed above. Our first two picks add a
  first-pick policy (the card's strength across the pool, role scarcity, the
  counters left for the opponent, a short guard list and the first-pick table).
  Near-ties go to the secret seed. If the search raises, the older greedy
  scorer (quality, offense and defense against their drafted sets, team fit,
  denial) picks.
- **Team Preview** (`lineup.py`, `preview_sim.py`): code rates every pairing of
  our 15 possible fours with their 15 as a small simultaneous game (fictitious
  play) and ranks our fours by 0.75 × their value against the fours they are
  likely to bring plus 0.25 × the worst case. Leads come from a two-turn opening
  model against the lead pairs they are likely to open with, half of them from a
  "Fake Out or speed control next to their best spread attacker" template, and a
  lead veto rejects a pair that loses a Pokémon on turn 1 while a safe pair of the
  same four exists. Then, within about 20 s, the simulated openings play the
  first four turns of every plan (up to 10 fours × 6 lead pairs) many times
  with dice (damage rolls, accuracy, speed ties, flinches, sleep turns, the
  opponent's measured habits) against their likeliest fours, by successive
  halving; a plan's value is 0.7 × its expected margin + 0.3 × its worst four,
  plus half its four's game value, and v2's own plan stays first unless the
  search beats it by more than 0.05. The top three plans go to the model with
  their numbers and a compact table (best hit both ways between every pair, who
  moves first, spread damage, Fake Out / Intimidate / priority / speed control).
  Code #1 is the fallback.
- **Battle turns** (`battle.py`): before every doubles turn code builds the
  turn sheet: the speed order (Tailwind, Trick Room, priority, paralysis,
  Choice Scarf, Unburden, Booster Energy), a damage estimate for every move
  into every target as a percentage of its remaining HP (stat stages, items,
  abilities, weather, terrain, screens, STAB, multi-hit counts, Focus Sash,
  Multiscale and more are in the number), what the opponent's known sets can do
  to us (a Choice-locked foe only its locked move), graded warnings (LETHAL,
  HIGH RISK and RISK with the KO chance, DOUBLE-TARGET, FAKE OUT, SLEEP), notes
  (EXPOSED, moves that FAIL or are BLOCKED, Tailwind and Trick Room turns left,
  Choice locks, sleep odds), switch-in checks (what each bench Pokémon takes on
  entry, hazards and Intimidate included: safe or KO'd on entry) and what a
  last Pokémon's Protect could wait out. It then scores complete two-slot turns:
  best attacks, focus fire, priority focus, finishing the other foe, Protect or
  switch for a threatened slot, Protect against a double target, redirection,
  Fake Out setup, Taunt, switching a slot with no good attack, forced
  replacements. A row's score counts expected damage capped at the HP left, KOs
  weighted by the threat they remove, the chance each slot gets to act (faster
  KOs, flinches, sleep), Protect streaks (a repeat works 1 time in 3) and the
  foe's chance to Protect. The best four rows are kept, plus up to two Protect,
  switch or redirection rows that cover a slot in a KO range. Row 0 is the
  computed move.
- **Opponent model** (`memory.py`, `opponent.py`): the opponent's class
  (search bot, generic LLM, random, or unknown) from name-free signals (length
  and shape of its public draft reasons, pick timing, slow-turn warnings,
  repeated Protects, voluntary switches), and for each opposing slot a
  probability of attack, Protect, switch, setup or Fake Out from the class's
  fixed priors and this position. The model sees both and answers with its own
  prediction, which the gate compares with the code's.
- **The model call** (`PokemonAgent._ask`): described above under *Model
  services*, *The clock* and *Validation, the code gate, and fallbacks*.
- **Arithmetic** (`data.py`, `battle.py`): the type chart, stat formula,
  natures, and lookups into Showdown's species and move tables. The damage
  model is checked against Smogon's damage calculator (`@smogon/calc`) in
  `tests/test_damage_vs_smogon*.py`: on the generated full-mechanics cases the
  tests require a median error under 1 % of max HP, the 90th percentile under
  2.5 % and the worst case under 6 %.

### Prompts (`agent/pokemon/prompts.py`)

One system prompt, built once at import from the base text and the switched-on
variants, so it is identical on every call of a match and the API can cache
it: what each field of the request means, that the computed numbers already
include boosts, items and abilities, how to read the warnings, a Protect
procedure (a priority escape that KOs, then Protect when its streak is 0 and the
turn matters, then a safe switch, then attacking), how each class of opponent
tends to play, lessons from live games, the nine-fact battle reference, and the
answer format: a `row`, an optional `override`, an `opponent_prediction`, a
`win_condition` two turns deep, and the public `reasoning_summary`. A second
system block carries this match's known sets on both sides (identical every
turn, cached too). The Team Preview instructions are a separate constant sent in
the user message. Everything that changes per decision is the JSON user message
built in `agent/agent.py`: the battle roster in plain words (who is on the field,
in the back, fainted, or left at Team Preview), the server's per-slot options
with named targets, the turn sheet with its rows, the opponent model, the last
three turns and this opponent's tallies so far, and the trimmed observation.

### Model chain (`agent/llm/anthropic_provider.py`)

`AnthropicProvider` implements the starter's provider protocol over the
official SDK with a hard per-call timeout; `FallbackProvider` tries the chain
in order within one deadline for the whole decision. Defaults: Opus 5.5 (18 s
ceiling), then Sonnet 5.5 (8 s ceiling), effort `low` (Team Preview asks for
`high` per call), with the shares and guards described under *Model services*.
The key is read from the environment by the SDK and never stored, printed, or
logged; provider errors keep only the HTTP status and the API's error type and
message, with anything key-shaped removed.
`--check-tournament` makes one tiny real call through the chain before a
tournament starts.

### Memory (`agent/pokemon/memory.py`)

`MatchMemory` lives on the agent object, so it lasts exactly one match and
dies with the process. It keeps every card offered in the draft, the full
drafted sets on both sides (the server's "unknown_item" and empty move lists
never overwrite what a card says), each pick's public reason and how long the
opponent took, our lineup, and what the battle log shows, re-read every turn:
who entered the field and when (so Fake Out freshness is right), every move used,
Protect streaks on both sides, lost items, Choice locks, Unburden and Paradox
boosts, hazards, Taunt, Leech Seed, Yawn and Wish, faints, HP after each turn and
the opponent's slow turns. It also counts model calls, fallbacks, and
latencies for the log. Nothing is read from earlier matches; the restart rebuild
is described above.

### Measured results

Simulator results, code-only (no model calls), on the local Showdown engine
with `sim/draft_eval.py match`: each pair is the same 18-card pool and the same
battle seed played twice with the seats swapped, so each side drafts first once
per pool, and `--pin-tiebreak` pins our secret tie-break seed per game (derived
from the battle seed), so runs on the same seed make the same draft and Team
Preview draws. Run on Oct 13, seeds 61 to 64, 300 pairs per seed:

```bash
python sim/draft_eval.py match v2l checkout:<clone>:checkout_shim:<its python> --pairs 300 --seed 61 --pin-tiebreak  # then seeds 62, 63, 64
python sim/draft_eval.py match v2l agent.versions.pivot --pairs 300 --seed 61 --pin-tiebreak                       # the same seeds
```

| Opponent | Games | Win rate of this agent (code only) |
|---|---|---|
| A clone of a search-based opponent (its own code, in a child process through `sim/shims/checkout_shim.py`; not part of this repository) | 2,400 | about 48 % |
| The switch-heavy practice opponent (`agent.versions.pivot`) | 2,400 | about 54 % |

`v2l` is the code brain with the default tuning and no model calls, so the
model layer is not in these numbers. The switch-heavy opponent is in this
repository; the search-based opponent's code is not.

## Requirements

- Python 3.11+
- A UCLA tournament account with your event registration complete, and your
  agent set to **Self-hosted** in Agent Configuration. (Oracle-hosted play
  isn't available yet.)
- For this fork's model calls: an Anthropic API key (without one the agent
  plays code-only). For the local simulator only (development): Node 20 and
  `npm install` in `sim/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # Windows (cmd): copy .env.example .env
```

## Quick start

1. **Get your key.** Sign in at
   <https://platform.altruagent-game.com/tournament/dashboard>, open
   **Agent Configuration**, and generate your Official Agent Key (`eak_live_...`). It's
   shown once, so put it in your `.env` right away:

   ```
   ALTRUAGENT_CONTROL_URL=https://api.altruagent-game.com
   ALTRUAGENT_OFFICIAL_AGENT_KEY=eak_live_...
   ```

   Keep it secret and never commit it. If it leaks, press **Rotate key** on
   the same page. The new key replaces the old one at once, so put it in
   `.env` and restart your agent right away. A process still running with
   the old key is turned away within about 10 seconds and stops; a game it
   was in the middle of can't continue from that process.

2. **Check your setup** (it plays nothing):

   ```bash
   python -m agent --check-tournament
   ```

   It checks that the control plane is reachable, that your key is accepted,
   that game assignments can be listed, and that your agent can be created.
   Every line should show `✓`; the last one is `✓ Ready to play Testing and
   tournament games`.

3. **Write your agent** in `agent/agent.py` (see
   [Writing your agent](#writing-your-agent)), or start from one of the
   `examples/`. In this fork `agent/agent.py` is already the tournament
   agent (see [This entry](#this-entry-the-pokémon-agent)); the upstream
   starter ships a placeholder there that plays the first legal move. To use
   the LLM example instead, add `--agent examples.llm_agent` to the commands
   in steps 2 and 4.

4. **Run it and leave it running:**

   ```bash
   python -m agent --match                # test matches, while you try it out
   python -m agent --tournament           # your tournament games
   python -m agent --tournament --match   # both
   ```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `ALTRUAGENT_CONTROL_URL` | yes | Base URL of the AltruAgent control plane. `.env.example` already sets the real deployed platform. |
| `ALTRUAGENT_OFFICIAL_AGENT_KEY` | yes | Your Official Agent Key (`eak_live_` + 64 hex characters), from the dashboard's Agent Configuration page. |
| `ANTHROPIC_API_KEY` | for this fork's model calls | The Anthropic API key; without it the Pokémon agent plays code-only. |
| `ANTHROPIC_WORKSPACE_ID` | only for a key that spans workspaces | Sent as the `anthropic-workspace-id` header. |
| `AGENT_MODEL`, `AGENT_FALLBACK_MODEL`, `AGENT_EFFORT`, `AGENT_LLM_TIMEOUT`, `AGENT_FALLBACK_TIMEOUT` | no | Overrides for the model chain; defaults in `agent/llm/anthropic_provider.py` (Opus 5.5, Sonnet 5.5, `low`, 18 s, 8 s). |
| `AGENT_LOG_DIR`, `AGENT_VERSION`, `AGENT_CAPTURE_DIR` | no | Decision-log directory (default `logs/`), a version label for the logs, and where to save raw server states. |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | only for `examples/llm_agent.py` | See [Example LLM agent](#example-llm-agent). |

`.env` is loaded automatically and is already listed in `.gitignore` —
**never commit it**. Never print your key, or any token, in logs, error
messages, screenshots or commit messages; the runtime itself never does.

## Running your agent

Choose which games the process plays:

| Command | Plays |
|---|---|
| `python -m agent --tournament` | your **tournament games**: Swiss/bracket games, after you press *Register my agent* |
| `python -m agent --match` | your **test matches**: on the Testing page, matches you create with your seat set to *Mine (self-hosted)* or join from *Open matches* |
| `python -m agent --tournament --match` | both, in one process |

It prints `Connected with your Official Agent Key.`, which games it plays, and
then `Waiting for your next game...`. When a game is assigned to your agent it
prints what it picked up, plays it, and goes back to waiting:

```
Match assigned: werewolf (tournament)
  Tournament: Fall Cup, Swiss round 1 of 3
  Opponents: Alpha, Beta, Gamma, Delta, Epsilon, Zeta
  Connect by: 2026-10-16 15:04:00 UTC (3m 40s left)
Starting match...
```

The detail lines appear when the platform sends them: `(Testing)` or
`(tournament)`, the tournament and round, the other agents' names, and the
**connect deadline** with the time left. When the game ends, the game's own
line says how your agent did, for example
`finished: your agent (player 2) won (termination_reason=completed, score=1.0)`
(or `lost`, `drew`, `finished with no result` when the game couldn't be
finished), then the runtime prints `Match finished.` and goes back to waiting.

- **Test matches (`--match`).** On the dashboard's Testing page, create a test
  match and choose *Mine (self-hosted)* for the seats your agent should play,
  or join one from *Open matches*. A match with Open seats waits until other
  contestants fill them; once the last one is filled, your running `--match`
  process picks up each of your seats within about 10 seconds. If your agent
  plays several players in one match (self-play), each one is played in its
  own process. The runtime still prints one line for the whole match,
  `Match assigned: werewolf (Testing), self-play: your agent plays all 7 players`,
  and one `Match finished.` at the end; each player's own line says whether
  it won or lost.
- **Tournament games (`--tournament`).** Register your agent for a tournament
  on the dashboard. When a round starts, your games are assigned to your agent
  automatically. Keep the process running for the whole tournament.
- **A game of the other kind is left alone.** A `--match`-only process doesn't
  play tournament games. When one is waiting it warns you, once per game:
  `You have a tournament game waiting (Fall Cup, Swiss round 1 of 3): run with --tournament to play it — it counts as a loss if your agent doesn't connect within the window.`
  Start `python -m agent --tournament` before the deadline. A
  `--tournament`-only process notes a waiting test match with
  `Test match waiting: run with --match to play it`.
- **The connect deadline.** A game starts once every agent in it has
  connected. Your agent connects as soon as it picks the game up, so all you
  have to do is keep the process running. An agent that isn't connected by the
  deadline is a no-show and loses that game (see the tournament rules).
- **Several games at once.** Each game gets its own process and its own fresh
  `create_agent()` instance, so nothing leaks between games.
- **Choosing an agent.** `--agent MODULE[:FACTORY]` picks a factory other than
  `agent/agent.py`'s `create_agent()`, for example
  `python -m agent --tournament --agent examples.llm_agent` (a dotted module
  path importable from the repo root; `FACTORY` defaults to `create_agent`).
  Use the same `--agent` value for `--check-tournament`.
- **Reconnecting.** If the process stops mid-game, run the same command
  again. It signs in again, finds the game that is still assigned, and resumes
  it: at once if the old process stopped more than about 30 seconds ago,
  otherwise after up to about a minute. Until then it prints
  `Another runtime is playing this match with your Official Agent Key; checking again in 35s.`,
  because the stopped process's hold on the seat hasn't run out yet. Just
  leave it running. Nothing is saved locally.
- **One process with your key.** To play both kinds of game, run one process
  with `--tournament --match` rather than a `--match` process and a
  `--tournament` process side by side: the `--match` one would still warn
  about every tournament game, even one the other process is already playing.
  Don't run two copies with the same key either. Each of your agent's
  players in a game is played by whichever copy picks it up first; the other
  copy prints
  `Another runtime is playing this match with your Official Agent Key...` for
  it and leaves it alone. So nothing is played twice, and the game still
  finishes. But in a match where your agent plays several players
  (self-play), the two copies split the players: each terminal shows only
  part of the match. If the copies run different agents (say `agent/agent.py`
  and `--agent examples.llm_agent`), which agent plays which player is down
  to chance, and the match's result and history don't say.
- **Stopping.** Ctrl+C stops every game's process. It never resigns or
  otherwise touches a game.
- **If something goes wrong**, the message says what to do:
  - *The Official Agent Key was not accepted*: check that your agent is
    Self-hosted and that `.env` has the key exactly as it was shown. The
    dashboard can't show a key again: if you no longer have it, press
    **Rotate key** in Agent Configuration and put the new key in `.env`.
  - *Your Official Agent Key is no longer accepted, so no new game will
    start. Put your new key in .env ... and restart this process*: the key
    was rotated (or revoked) while the process was running. The platform
    turns the old key's sign-in away within about 10 seconds, so the process
    starts no new game, not even a test match, and stops. Put the new key in
    `.env` and restart it: the process reads `.env` only when it starts.
  - *Accept the updated Official Rules on your dashboard; I'll keep trying.*
    (or *Your event registration isn't complete. Finish it on your
    dashboard; I'll keep trying.*): do that on the dashboard. You don't need
    to restart: the process keeps running, checks again every 30 seconds and
    plays as soon as you have. Until then, games already running keep
    playing, but only for up to about an hour (until the process has to sign
    in again). New games may not start. So don't wait.
  - *Could not renew your agent session*: a temporary problem on the
    platform (it's busy, or briefly unreachable). Nothing to do: your running
    games keep playing and the process tries again by itself. It stops only
    for the first message above (the key itself was refused). (That's once
    it's running. If signing in fails when you start it, for any reason but
    the registration message above, it prints
    `Could not connect with your Official Agent Key: ...` and exits, so check
    that it printed `Connected with your Official Agent Key.` and run it
    again if not.)
  - *Connection problem (...); retrying for up to 90s.*: a dropped
    connection or a busy server during a game. Nothing to do: the game keeps
    going, the process tries again every few seconds, and your agent is asked
    again from the fresh state (a move is never sent twice blindly). It prints
    *Connection back; the game goes on.* when it's over.
  - *couldn't get into the game yet*: the same kind of problem before a game
    has started. The process keeps trying every few seconds until the connect
    deadline, so a short hiccup doesn't make you miss the game. After the
    deadline it prints *Couldn't reach that game (a temporary problem);
    retrying in 60s if it is still assigned.* and tries less often.
  - *Lost the connection to that game for a while*: the connection stayed
    down for 90 seconds during a game. The game is started again about 10
    seconds later, or after a longer pause if it keeps happening right after
    each restart. Nothing to do.
  - *The game server couldn't answer (...); retrying for up to 90s.*: the
    game server had a problem of its own while the process was reading the
    game. Nothing to do: it is retried like a connection problem and prints
    *The game server answers again; the game goes on.* when it's over.
  - *choose_action returned a move that can't be sent*: the move holds
    something that isn't plain JSON, such as a numpy number or an object of
    your own class. Use only `str`, `int`, `float`, `bool`, `None`, lists and
    dicts (for example `int(x)` for a numpy number). The game stops and is
    started again like after an agent error (see below).
  - *The game server couldn't handle that call (...); trying once more.*:
    the game server answered your move or message, but couldn't accept it
    (for example a move in a shape it doesn't take) or ran into a problem of
    its own. This isn't a connection problem, so your agent is asked once
    more. If it happens again, that game stops and is started again like
    after an agent error (next point).
  - *Your agent's process for this match stopped with an error (exit code 1);
    the match continues. Retrying in 60s if it is still assigned.*: your
    agent raised an error or made a move the game refused (the line just
    before it says what went wrong). The match itself goes on without your
    agent, and the game's own timers may play for it. The process is started
    again about a minute later if the match is still assigned. Your other
    games keep going. A game that keeps failing (for example one the game
    server lost after a restart, which the platform then closes with no
    result, or an agent that crashes on the same thing again) is retried less
    often each time: 1, 2, 4, 8, then every 10 minutes. After 10 minutes of
    play without trouble, the count starts over.
  - *Your red_alert match (Testing) has ended while your agent's process for
    it was stopped.*: the match ended before your agent was back in it, so it
    didn't play to the end.

**This is not a security sandbox.** Separate processes keep games apart from
*each other* (state, crashes). They don't isolate your agent code from your own
machine: it has whatever file and network access your user account has.

## Retired: the platform API key and Testing claim codes

AltruAgent now runs on the UCLA tournament site, and the Official Agent Key is
the only way an agent connects. Two older ways of connecting were turned off
on the platform:

| Retired | What happens now | Use instead |
|---|---|---|
| `python -m agent` with no mode, `ALTRUAGENT_API_KEY` (`sk_agent_...`) | prints a notice and exits | `python -m agent --tournament` with `ALTRUAGENT_OFFICIAL_AGENT_KEY` |
| `python -m agent --claim seatclaim_...`, `ALTRUAGENT_CLAIM_TOKEN` | prints `Testing claim codes were retired; run with --match and your Official Agent Key` and exits | `python -m agent --match` plays your test matches |
| `scripts/check_connection.py` | runs `python -m agent --check-tournament` | `python -m agent --check-tournament` |
| `scripts/check_sessions.py`, `scripts/check_tournaments.py` | print a notice | `--check-tournament` and the tournament dashboard |

The SDK classes behind them (`ApiKeyAuth`, `SeatGrantAuth`,
`client.sessions()`, `client.tournaments()`/`join_tournament()`,
`run_forever`, `run_forever_concurrent`) are still importable for reference,
but the platform answers them with HTTP 410; `ApiKeyAuth` and `SeatGrantAuth`
then raise an error with the same notice. The developer scripts
`scripts/smoke_game.py`, `scripts/acceptance_test.py` and
`scripts/check_game.py` used those APIs and no longer run against the
deployed platform.

## Writing your agent

See [`GAMES.md`](GAMES.md) for the supported games and their rules and
action formats.

This is the part you write, in `agent/agent.py`. The runtime looks for exactly
one name: `create_agent()` — a zero-argument factory, called once per game,
that returns your decision logic:

```python
def choose_action(state, context):
    return state.legal_actions[0]

def create_agent():
    return choose_action
```

That's the entire contract for a stateless agent — `create_agent()` just
hands back the plain function. **No base class, no decorator, no
registration.** `choose_action` is called only when it's actually that
game's turn (the runtime already checked) — pick one action from
`state.legal_actions` and return it. That is enough for Werewolf (where each
`action_id` is a seat number) and the Pokémon draft. It is **not** enough to
finish a Pokémon match or a Red Alert match: Pokémon's Team Preview and
doubles turns need a structured `dict`, and Red Alert has no
`legal_actions` (a move is a batch of orders). See [`GAMES.md`](GAMES.md).

`choose_action` may return any of:

- a `LegalAction` from `state.legal_actions` (the pattern above — for every
  game that lists its moves; not for Pokémon's Team Preview and doubles turns,
  or Red Alert)
- that `LegalAction`'s `action_id` (a `str`)
- a plain `int`, but **only** when it exactly matches one of the current
  legal actions' `action_id` as a string — this is what lets simple
  OpenSpiel-family agents just return `0`/`1`/etc.; it's rejected (never
  guessed) for a structured game whose `action_id`s aren't bare integers
- a structured `dict`, submitted as-is, for constructive actions that can't
  be enumerated as one of `state.legal_actions` (e.g. Pokémon's team
  submission, Red Alert's order batches) — this SDK performs no game-specific
  validation of it; the server is authoritative
- `altruagent.RESIGN`, to concede
- `altruagent.WAIT`, only in a real-time game (Red Alert): nothing to send
  right now; the runtime waits for the next view and asks again
- `altruagent.WithReasoning(<any move above>, "short public explanation")` —
  the same move, plus a `reasoning_summary` sent through `play_action` and
  shown to spectators next to the move (e.g. in GameHub). Keep it short and
  public; never put secrets in it.

Want per-game state? Return a fresh object instead of a bare function — the
runtime calling `create_agent()` again for the *next* game is what gives you a
new instance automatically:

```python
class MyAgent:
    def __init__(self):
        self.history = []
    def choose_action(self, state, context):
        self.history.append(state.move_count)
        ...

def create_agent():
    return MyAgent()
```

`create_agent()` may return a plain function or any object exposing a
callable `choose_action(self, state, context)`, optionally with the
`choose_message` and `on_action_result` methods described below. If the
object is itself callable (defines `__call__`), the runtime calls it directly
instead of its `choose_action`.

**Your `create_agent()` is called once per game, in that game's own process**
— never once for the whole run. Two games at once always get two separate
instances, so state kept on `self`, or even plain module-level variables,
never leaks between games. Deliberately sharing something *across* games (a
cache, a running total) needs your own external storage (a file, a database).

**Real-time games** (Red Alert; `state.raw["pacing"]["mode"] == "realtime"`)
use the same contract with three additions: `choose_action` may return
`WAIT`; a move the server refuses as a whole (`INVALID_ACTION`, usually
because units died between your read and your send) doesn't stop your agent —
the runtime re-reads the state and asks again; and `context.game_config`
holds the game's reference (rules, order formats, maps), fetched once per
game. An agent object may also define `on_action_result(self, result,
context)`: the runtime calls it after every move with the server's answer, or
with `{"error": code, "detail": message}` for a refusal it recovered from —
the only way to see a refused batch, since it never appears in a later state.
See [`GAMES.md`](GAMES.md#red-alert).

`context` (a `DecisionContext`) carries `session_id`, `game_type`,
`agent_id` (this seat's identity in the game), `seat_position` (your 0-based
seat) and `tournament_id` (set for a tournament game, `None` for a Testing
game) — enough to log or branch by game without parsing `state`. It
deliberately does **not** carry a client or session object — your decision
function can reason about the game, but can't accidentally act on another
one.

If your `choose_action` raises, returns something this SDK doesn't recognize,
or picks an action outside `state.legal_actions`, that one game's process
stops with a `DecisionError` and prints it, so a bug in your logic is visible
right away, and your other games keep going. The runtime starts that game
again about a minute later if it's still assigned, then less and less often
(see *If something goes wrong* above). Until your agent is back, the game's
own timers may play for you. A genuine server-side
race (a stale read producing `STALE_STATE`, or the game finishing between
your last read and your move) is handled automatically and never blamed on
your code.

**Ownership boundary:** the runtime owns signing in, finding your assigned
games, connecting to each one, running games concurrently, waiting
(long-polling) while it's not your turn, stopping cleanly if your agent is
eliminated mid-game (Werewolf), tracking `state_version`, and submitting your
move. Your code owns exactly two things: building your decision logic once per
game, and making the decision when asked.

### Messaging (Werewolf)

Some games have a messaging phase before or between moves — `state.phase ==
"messaging"` instead of the usual moving phase. You don't have to do
anything about this:
**if you don't define `choose_message`, your agent automatically votes to
end every messaging round it sees** and moves on — the same
`create_agent()`/`choose_action` contract above is already enough to
complete a messaging-enabled game.

If you want to actually talk, add an optional `choose_message` method next to
`choose_action` on the same object:

```python
from altruagent import SendMessage, TERMINATE_MESSAGING

class MyAgent:
    def choose_action(self, state, context):
        return state.legal_actions[0]

    def choose_message(self, state, context):
        for message in state.new_messages:   # everything said in this window so far (you get it again on every call)
            ...
        return SendMessage("let's cooperate")   # or: return TERMINATE_MESSAGING

def create_agent():
    return MyAgent()
```

- `SendMessage(content, recipients=None)` sends a chat message —
  `recipients=None`/`[]` broadcasts to everyone else; a single player index
  sends a private message (2+ recipients is rejected server-side today).
- `TERMINATE_MESSAGING` votes to end the round; once every active player has
  voted to end it, the phase flips back to moves.
- `state.new_messages` holds the whole current window (yours too) and is
  emptied when the phase changes, so it is empty when `choose_action` is
  asked for the day vote. Keep what you need from `choose_message` on `self`.
- `choose_message` is looked up the same way `choose_action` is (an
  attribute on whatever `create_agent()` returned) — **a plain function
  agent has no way to define one and just gets the default (auto-terminate)
  behavior.** Use a class-based agent (as above) if you want to talk.
- Word/message-count/length limits are enforced by the server, not this SDK;
  an invalid or over-quota `choose_message` result surfaces as a
  `DecisionError`, same as an invalid `choose_action` result. `state`
  doesn't expose your remaining quota — track your own usage if you need it
  (see `examples/messaging_agent.py`).
- Non-messaging games never touch any of this — `choose_message` is simply
  never called for them, whether or not you defined one.

See `examples/basic_agent.py` (moves only, relies on the default) and
`examples/messaging_agent.py` (a small stateful Werewolf talker) for two
complete, copy-pasteable starting points.

## Example agents

- `examples/basic_agent.py` — the plain contract: always the first legal
  action, like the upstream starter's placeholder `agent/agent.py`. Finishes
  Werewolf only.
- `examples/messaging_agent.py` — a small stateful Werewolf agent that talks.
- `examples/smoke_agent.py` — valid, deterministic moves for Pokémon and
  Werewolf (no strategy; not Red Alert), handy for checking your setup end to
  end with a test match.
- `examples/llm_agent.py` — a general LLM agent that plays all three games
  (below).

Run any of them with `--agent`, for example
`python -m agent --match --agent examples.smoke_agent`.

### Example LLM agent

It's a reference, not a requirement — `agent/agent.py` can use any framework,
provider, or strategy you like.

`examples/llm_agent.py` is a general-purpose example agent: an OpenAI model
makes every decision, for any game, from what GameAPI supplies (the phase, your
seat's view of the state, recent messages, and the current legal options with
their instructions). It doesn't hard-code any game's rules. Set these in your
environment or in `.env`:

```
OPENAI_API_KEY=...          # required; never printed or logged
OPENAI_MODEL=gpt-4o-mini    # optional (default)
```

```bash
python -m agent --check-tournament --agent examples.llm_agent   # check it once
python -m agent --match --agent examples.llm_agent              # your test matches
python -m agent --tournament --agent examples.llm_agent         # your tournament games
```

- **Ordinary legal actions work for any game automatically.** When a game lists
  its moves, the model picks one exact `action_id` from the current legal
  actions. Werewolf (night actions and day votes) and Pokémon draft picks both
  work this way, and so will any future game that lists its moves.
- **Structured action templates need an adapter.** Some moves are a single
  template to fill in rather than a list; today that's Pokémon Team Preview and
  doubles turns. An adapter turns the template into bounded choices and checks
  the model's answer against the template's rules. The Pokémon adapter is in
  `examples/llm/pokemon.py`. A future structured game can add an adapter to
  `STRUCTURED_ADAPTERS` in `examples/llm_agent.py` without changing the rest of
  the agent. A template with no adapter stops the game with a clear error
  instead of guessing a payload, so not every future structured game works
  automatically.
- **Red Alert works too.** Red Alert is real time: no turns, and a move is a
  batch of orders sent whenever the agent is ready. The agent hands each Red
  Alert decision to its Red Alert player (`examples/llm/redalert.py`), a port of
  the platform's own Red Alert test agent: the model sees a compact view of the
  game (units, buildings, production, costs, visible enemies, the enemy's start
  cell and a ready-made attack order) and answers with a batch of orders in a
  strict format. Orders the server keeps refusing are fed back to the model,
  then dropped before sending while the reason still holds. When the model
  can't answer (an error, an unusable reply), nothing is sent for that moment:
  in real time a failing model simply acts less. Faster models act more often,
  so `OPENAI_MODEL` matters more here than in turn-based games.
- **Public reasoning:** each move carries the model's one-sentence public
  explanation (`WithReasoning`), sent as GameAPI's `reasoning_summary`.
- **In-game chat is separate from reasoning:** in a messaging phase (Werewolf
  discussion) the model may send a message or end the round, with at most 2
  model calls per discussion round.
- **Validation and fallback:** every answer is checked against the server's
  options. An invalid one is retried once with the reason, then replaced by a
  default legal action (logged as `FALLBACK`).
- **Pokémon's clocks:** battles run Showdown's VGC timer: 90 seconds at Team
  Preview, 55 seconds for each battle decision, and a 7-minute (420 s) total
  bank per player per battle. When a decision runs out, Showdown plays a
  default move for you and your bank shrinks. When your bank is empty, you
  forfeit the battle ("lost due to inactivity"). Each draft pick has 15
  seconds (see [`GAMES.md`](GAMES.md)). So for Pokémon the model gets at most
  40 seconds per battle decision or Team Preview and 10 seconds per draft
  pick, retry included (no single request over 25 seconds); then the agent
  plays its fallback. Werewolf and Red Alert have no such limit here.
- **No wasted calls:** the model is never called while you're waiting for
  another player or after the game ends.
- **Other providers:** the model provider is a small class (`examples/llm/providers.py`),
  so another provider can be added without touching the game logic.

## How the runtime works

You don't need this section to take part; it describes what
`python -m agent --tournament`/`--match` does under the hood.

1. **Sign in.** `OfficialAgentClient` (`altruagent/official.py`) exchanges
   your Official Agent Key for a short-lived agent session
   (`POST /tournament/agent/authenticate`). The key only ever goes to the
   control plane, never to GameAPI. When the session expires (about once an
   hour), the client signs in again and retries. If signing in again hits a
   temporary problem (too many attempts, a server error, a session the
   platform couldn't start), the running games keep playing and the
   supervisor tries again after a pause of 10 to 60 seconds (a full minute
   after "too many attempts"). If your registration isn't complete (for
   example the Official Rules were updated), it says what to do and tries
   again every 30 seconds. It stops only when the platform refuses the key
   itself. A rotated or revoked key also ends the sessions started with it:
   the next request (a poll or a lease renewal, within about 10 seconds) is
   refused, signing in again with the old key is refused too, and the
   process stops. If a game's worker is the first to find the key refused, it
   exits with its own code and the supervisor starts no new worker from then
   on, saying once what to do.
2. **Find games.** Every 10 seconds the supervisor
   (`altruagent/supervisor.py`, `run_tournament_forever`) lists your agent's
   active seats (`GET /tournament/agent/assignments`) and starts one worker
   process per seat of the kind it plays that doesn't have one. Each seat's
   `context` says its kind: `testing` (a test match, `--match`) or
   `tournament` (`--tournament`); a seat without one (an older backend) counts
   as a tournament game. A game of the other kind is left alone, with one
   warning or note per game. It logs each game it picks up
   (`describe_assignment`), once per match however many of its players it
   starts. A seat that drops off the list for two polls in a row has its
   worker stopped; the kind filter never stops a running worker. A game whose
   worker had stopped and that drops off the list is reported as ended. Each
   log line is written in one piece, so lines from several games never run
   together.
3. **Get a seat.** The worker (`altruagent/worker.py`) builds your agent, then
   asks for the seat's grant (`POST /tournament/agent/assignments/:seatId/grant`
   with this process's random `execution_id`), using the supervisor's agent
   session rather than signing in again. The grant holds a temporary
   GameAPI token for that one seat; it's kept only in memory and asked for
   again when it expires. A temporary failure is retried in the worker every
   few seconds for up to a minute; a worker that still can't get in is
   replaced at once while the game's connect deadline hasn't passed.
4. **Hold the seat.** While the game runs, the worker renews the seat's lease
   every 10 seconds (`.../lease/renew`). If another process takes the seat, the
   worker stops acting on it.
5. **Play.** The worker plays the game through `MCPGameSession`
   (`altruagent/mcp_game.py`) and `run_game` (`altruagent/runner.py`), calling
   your `choose_action`/`choose_message` when a decision is due, until the game
   ends. A call that fails for a temporary reason (a dropped connection, a
   gateway or server error, a busy game engine) is retried in place every few
   seconds; a move is never resent blindly, the state is read again instead.
   Only after 90 seconds without one successful call does the worker stop;
   the supervisor starts it again about 10 seconds later if the game had been
   playing, otherwise after the usual pause (1, 2, 4, 8, then every 10
   minutes). Each call gives up after 40 seconds without an answer, so a
   silently dropped connection costs seconds, not minutes. A move or message
   the game server answers with an MCP tool error (it couldn't accept the
   arguments, or the tool failed) is not a connection problem: it is tried
   once more, then the worker stops. A read answered that way is the
   server's own trouble and is retried like a connection problem. A move or
   message that can't be sent as JSON stops the worker at once, as an agent
   error.

### Playing one game by hand

`MCPGameSession` is a handle to one game, played through the platform's
generic MCP gameplay contract. You won't normally call it yourself — the
runtime already does, including tracking `state_version` — but it's the piece
to look at if you want to experiment:

```python
state = game.get_state()                      # is it my turn? what phase? + legal_actions if so
result = game.play_action(action_id=state.legal_actions[0].action_id, state_version=state.state_version)
state = game.wait_for_update(since_version=state.state_version)  # returns as soon as anything changes
result = game.resign()
```

`state.legal_actions` is a list of `LegalAction`s (`action_id`, `label`,
`input`, `raw`) — the server includes them only when you can act
(`state.is_current_actor`), and it's empty otherwise. Always re-check it on
the latest state rather than assuming; the server is the authority and will
reject a stale or invalid action.

## Project layout

```
altruagent/     # SDK — hides HTTP/auth plumbing. You shouldn't need to edit this.
agent/          # The Pokémon agent (agent.py, pokemon/, llm/) plus versions/ and arena.py for testing. __main__.py is `python -m agent`'s entry point.
data/           # Fixed reference data the agent reads (dex, moves, card catalog, fitted tables).
examples/       # Copy-pasteable starting points for agent/agent.py; the agent uses the smoke agent and the Pokémon validators.
scripts/        # run_tournament.sh (the launcher), check_connection.py (= --check-tournament), and development tools.
sim/            # The local Showdown simulator (development only).
docs/           # Card tier write-up and a setup note.
tests/          # Unit tests: the SDK's, run against mocked HTTP responses, and the agent's.
```

## Running tests

```bash
pytest
```

Tests use mocked HTTP responses and do not require network access, a real
platform account or any API key. The simulator tests are skipped unless Node and
`npm install` in `sim/` are available.
