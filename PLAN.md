# PLAN.md — UCLA AI Agent Gaming Tournament: Pokémon VGC Doubles Draft agent

Approved by James on Oct 6, 2026, before any agent code was written.

Dates as planned on Oct 6: plan written **Tue Oct 6, 2026**; code freeze **Tue Oct 13**; tournament **Fri Oct 16,
10:00–12:00 PDT**.

---

## Final state at submission

**What is deployed.** Commit `b74574a` (deploy #9), the code that competes, plus documentation commits on top of
it that change only `README.md` and `PLAN.md`. Commit IDs in the diary below predate a history cleanup on Oct 15
(old notes reworded); the code did not change. The submitted version is the latest commit on the repository's
default branch at the Submission Deadline. README.md ("This entry: the Pokémon agent") describes the agent in full.
Everything below this section is the plan and build diary from Oct 6 on: where it describes an earlier design,
model or default, this section is what plays.

**Main components**
- `agent/agent.py`: `PokemonAgent` routes each request (draft pick, Team Preview, battle turn), builds the model's
  message, budgets the clock, retries once on an invalid answer, runs the lethal recheck and the row gate, falls
  back to the computed move, plays the starter's smoke move on an exception, and writes the decision log.
- `agent/llm/anthropic_provider.py`: the model chain. **Claude Opus 5.5 (`claude-opus-5-5`) makes the model
  decisions; Sonnet 5.5 (`claude-sonnet-5-5`) answers when Opus errors or times out; then the computed move plays.**
  Timeout ceilings 18 s and 8 s plus a 2 s wall-clock guard; effort `low` for battle turns and `high` for Team
  Preview (per call); strict JSON output; `max_tokens` 4,000; no `thinking` parameter; SDK retries off.
- The clock (`agent/agent.py` `_clock_budget`): the server reports about 50 s per battle decision, 85 s at Team
  Preview and a 420 s bank. Model deadline = min(decision time left − 8 s, (bank − 90 s) ÷ turns still needed), at
  most 20 s a turn; at Team Preview what the ~20 s search leaves, at most 60 s, and code #1 under 10 s. A deadline
  under 6 s skips the model; a second call needs 6 s left; two all-failed decisions in a row skip the model once.
- `agent/pokemon/draft.py`, `matchup.py`, `counterpick.py`: the code-only, counter-aware draft search (no model call).
- `agent/pokemon/lineup.py`, `preview_sim.py`: Team Preview from the four-against-four game, the lead veto and
  ~20 s of simulated openings; the top three plans and a matchup table go to the model.
- `agent/pokemon/battle.py`: the turn sheet and its scored rows (the best four plus up to two Protect / switch /
  redirection rows that cover a slot in a KO range); row 0 is the fallback.
- `agent/pokemon/opponent.py`, `memory.py`, `match_state.py`: the opponent's class and per-slot prediction; match
  memory from the draft and the battle log (re-read every turn), rebuilt from `data/cards.json` after a restart;
  per-match caches emptied at a match's start.
- `agent/pokemon/prompts.py`: the system prompt (with `prompt_v2` and `prompt_kb`), the known-sets block, the Team
  Preview instructions.
- `agent/pokemon/tuning.py` and `data/`: the weights and switches; the dex and move tables, the 48-card catalog,
  the fitted team model, the per-card and first-pick tables, the tier list.
- `scripts/run_tournament.sh`: the launcher (pre-flight with one real model call, detached restart loop, runtime
  log, raw-state capture to `tests/fixtures/live/`).
- In the simulator tools the player spec `fable` is the full agent (`agent/agent.py` with the model chain from
  `.env`, Opus 5.5 by default); the name is historical.

**Default-on switches and the weights that change behaviour** (`tuning.DEFAULTS`; every other number there is a
plain weight)
- `draft_v2` 1: the counter-aware draft search instead of the greedy scorer.
- `draft_lookahead` 3, `draft_counter_replies` 3, `draft_minimax` 0.5, `draft_reply_depth` 1: their next pick tried
  with three team-model and three counter-picker replies, the worst reply weighted half.
- `draft_four_eval` 1, `draft_their_fours` 3: a finished draft valued as our four against their three likeliest fours.
- `draft_card_w` 1: the team model's fitted per-card strengths.
- `draft_counter_prior` 0.02: the tier rating as the counter-picker's strength prior.
- `draft_first_pick_w` 1 (strength 1, scarcity 0.5, exposure 0.2, guard 0.1, table 1): the first-pick policy, full
  at our first pick, half at our second.
- `draft_hole_w` 0.5, `lineup_hole_w` 0.5: a coverage-hole penalty in the draft value and at Team Preview.
- `draft_support_w` 0.25: a counter blunted by Intimidate or a safe switch-in partner counts for less.
- `draft_card_adjust_w` 0.5, `draft_card_adjust_them_w` 0.5: the per-card table in the finished-draft value, with
  the Archaludon floor.
- `draft_tiebreak_eps` 0.01, `lineup_tiebreak_eps` 0.01: near-ties (within 1 % win chance) drawn by a secret
  per-match seed.
- `lineup_v2` 1 (`lineup_v2_robust` 0.25, `lineup_v2_template` 0.5): Team Preview from the four-against-four game.
- `lead_veto_w` 0.5: no lead pair that loses a Pokémon on turn 1 while a safe pair of the same four exists.
- `lineup_sim_budget_s` 20 (with the other `lineup_sim_*` keys): simulated openings re-rank the plans.
- `lineup_min_model_s` 10: the Team Preview model call needs 10 s left after the search.
- `hp_units` 1: damage valued in % of max HP, capped at what the target has left.
- `choice_lock_rerank` 1: a Choice holder is steered off moves dead against a remaining foe.
- `focus_risk` 0.35, `opp_protect_risk` 0.3, `fake_out_use` 0.5: double-target risk, a foe's Protect chance in KO
  range, a fresh opposing Fake Out used half the time.
- `ko_threat_w` 0.5, `exposure_w` 1: a KO is worth more the bigger the threat removed; self-exposing moves are charged.
- `sleep_future_w` 0.5, `sleep_use_prior` 0.6: sleep threats priced.
- `foe_sleep` 1: a sleeping, paralysed or frozen foe's threat counts only as often as it acts.
- `choice_protect_guard` 1: a Protect that would lock an unlocked Choice item earns nothing in the rows and is
  refused as an override.
- `hp_power_first` 1: Eruption / Water Spout are credited at the HP their user likely has when it moves (after
  the hits of foes that move first); a KO counts as guaranteed only if it still lands after those hits.
- `lone_foe_split` 1: with only one of theirs able to act, its single-target hit lands on one of our slots, not
  both; its KO-before-we-move counts at each slot as often as it aims there.
- `speed_end_stall` 1: a Protect is credited with the first strike it buys when Trick Room / their Tailwind ends
  after this turn.
- `gate_repeat_protect` 1: the row gate never forces a repeat Protect (1 in 3) over the model's listed row.
- Off (tested Oct 16, not shipped): `foe_aim_split` (+9.7 vs a reader bot, -1.5 vs the others), `mix_play`,
  `dodge_adapt`, `preview_speed_mode` (-5.3 vs the search-bot clone).
- `override_margin` 0.15: the row gate (a lower row or an override plays only within 15 % of row 0, or an override
  whose opponent prediction matches the code's).
- `recheck_v2` 1: the lethal recheck only where row 0 Protects or switches the slot, pointing the model to row 0.
- `prompt_v2` 1: the Protect procedure treats DOUBLE-TARGET as a risk, lessons from live games, the prediction
  vocabulary in the code's words.
- `prompt_kb` 1: the nine-fact battle reference in the system prompt.
- Off: `reply_mixture` (and `answer_rules_v2`, which needs it), `protect_prior_rows`, `draft_tier_adjust`,
  `draft_sleep_w`, `lineup_sleep_w`, `draft_tiebreak_exposure`, `draft_mode_w`, `draft_role_w`,
  `draft_deny_mode_w`, `lineup_answer_w`, `lead_priority_multihit_w`, `lead_ohko_w`, `lineup_predict_opp`,
  `lineup_v2_lead_robust`, `lineup_v2_four_template`.

**Last measurement** (Oct 13, code only, `sim/draft_eval.py match … --pin-tiebreak`, seeds 61–64, 300 pinned pairs
per seed): about 48 % of 2,400 paired games against a clone of a search-based opponent, about 54 % against the
switch-heavy practice opponent (`agent/versions/pivot.py`).

---

## 0. Context (why this plan looks the way it does)

- You chose **Pokémon Showdown only**. Werewolf is dropped entirely.
- The tournament's Pokémon game is `pokemon_vgc_doubles_draft`: two agents snake-draft 6 Pokémon
  each from a shared 18-card pool, pick 4 of 6 at Team Preview, then play a 4v4 doubles battle.
- Decisions by **Claude Opus 5.5** (`claude-opus-5-5`) through the official `anthropic` Python
  SDK, key in `.env` as `ANTHROPIC_API_KEY`. Fallback chain: Sonnet 5.5, then a code-only move. (The Oct 6 plan
  named Fable 5.1; Opus 5.5 replaced it that night, §5d.)
- Guiding rule from you: **reliable beats clever.** Every decision has a pure-code answer that is
  legal and sane, and the LLM only ever *upgrades* it. The agent can never crash a match because
  the model was slow, down, or wrong.

### What I already did this session
- Forked `UCLA-Trustworthy-AI-Lab/altruagent-starter` to **github.com/holmjames/altruagent-starter**
  and cloned it to `Claude Code Projects/altruagent-starter/` (`origin` = your fork, `upstream` = UCLA).
- Read README.md, GAMES.md, agent/agent.py, examples/llm_agent.py, examples/llm/pokemon.py,
  examples/llm/providers.py, examples/llm/base.py, altruagent/models.py, and the platform docs
  (skill.md, skill/pokemon, skill/04-tournaments, skill/official-agent, join page, agent guide).

---

## 1. The contract, in plain language (what your code is and what it receives)

Your entire job is one file, `agent/agent.py`, exposing one function: `create_agent()`.

- The runtime (`python -m agent ...`) logs in with your Official Agent Key, polls for games every
  ~10 s, and for **each game** starts a separate process that calls `create_agent()` once. Whatever
  that returns is "your agent" for that one game. Returning an object (not a bare function) gives
  you **per-match memory for free**: fields on `self` live exactly as long as the match.
- The runtime then calls `your_agent.choose_action(state, context)` **only when it is actually your
  turn**. You never poll, never track turn order, never talk to the server. You look at `state`,
  return one move, done.
- `state` (a `GameState`, `altruagent/models.py`) has: `phase` (`draft` / `team_preview` / `moving`),
  `legal_actions` (list of `LegalAction` with `action_id`, `label`, `input`), `observation`
  (the game's text/JSON view for your seat), `state_version`, `is_terminal`, and `raw` (the complete
  server payload, where all the Pokémon detail lives).
- `context` (a `DecisionContext`) has `session_id`, `game_type`, `agent_id`, `seat_position`.
- What you return, per phase:
  - **Draft**: one of `state.legal_actions` (ids look like `draft_pick:<card_id>`). Just return the
    `LegalAction` object.
  - **Team Preview**: a dict `{"type": "select_lineup", "bring": [4 species], "leads": [2 of those]}`.
  - **Battle turn**: a dict `{"type": "doubles_turn", "slot_0": {...}, "slot_1": {...}}`, each slot a
    `move` (with `move_id` and `target`), a `switch` (with `species`), or `pass` (only when offered).
    Targets: `1`/`2` = opponent positions, `-1`/`-2` = your own slots, `0` = no target.
  - Optionally wrap any of these in `WithReasoning(move, "one public sentence")` so spectators see why.
- `choose_message` exists for chat games. Pokémon has messaging disabled, so **we do not implement it**.
- If `choose_action` raises or returns something illegal, the runtime raises `DecisionError` and that
  match's process exits (it is retried after a cooldown, and the server auto-plays a random move for
  you when the clock runs out). Our design makes this unreachable: every return value passes the validators in
  `examples/llm/pokemon.py` before it leaves our code.

---

## 2. What the docs say that shapes the plan (verified Oct 6)

| Topic | Fact | Consequence |
|---|---|---|
| Draft clock | **15 s per pick**, measured from when it becomes your turn and *including* any model call. Timeout = server picks a random card; late picks are rejected `STALE_STATE`. | **Draft is pure code.** No LLM call in the draft, ever. A model call can take 5–20 s. |
| Battle clock | ~~300 s~~ **Corrected Oct 7 (upstream GAMES.md):** Showdown's VGC timer. **90 s** Team Preview, **55 s** per battle decision, **420 s total bank** per battle; an empty bank is a forfeit. | The server reports about 50 s left per battle decision and 85 s at Team Preview when it asks. Each decision gets one model deadline from that clock: min(decision left − 8 s, (bank − 90 s) ÷ turns still needed), at most 20 s a turn (60 s at Team Preview); Opus ≤ 18 s and Sonnet ≤ 8 s inside it (+2 s guard each); a second call only with ≥ 6 s left; under 6 s the computed move plays. Median Opus call is ~3 s. |
| Draft info | Pool cards are **complete sets**: species, item, ability, nature, EVs, 4 moves. `rosters` and `picks` are public to both players. | We know the opponent's **exact** 6 builds before the battle starts. Most LLM-only agents only use what the battle "reveals". This is our biggest edge: store every drafted card in match memory. |
| Item Clause | Your six must hold six different items; clashing cards are simply not offered. | Nothing to enforce; `legal_actions` already filters. |
| Team Preview | You see both full rosters with `types` and `base_stats`; bring 4, lead 2. | Lineup is chosen against the opponent's known 6 (and their known moves from the draft). |
| Battle observation | Per slot: `available_moves` (`id`, `type`, `category`, `base_power`, `accuracy`, `priority`, `current_pp`, `targets`, named `target_options`), `available_switches`, `force_switch`; `team`/`opponent_team` with HP fraction, status, boosts, types, base_stats, revealed moves; `weather`, `field`, `turn`. No Terastallization. | Enough to compute type effectiveness, rough damage and speed order in code. |
| Illegal move | `INVALID_ACTION`, recoverable; not a forfeit. | Still avoid it: validators from `examples/llm/pokemon.py` run before every return. |
| Tournament format | **Swiss, 6 rounds** (win +1, loss 0, draw 0; byes +1), then **top-cut 32 single-elimination bracket, best-of-3**. Tiebreak: points, then Buchholz, then random. Draws/no-results replayed in bracket. | Expect 6–11 games on the day, one at a time. Consistency matters more than peak play. |
| Connect window | **4 minutes** from game creation (connection = first gameplay call). Miss it = loss; the connected agent gets +1 free. | `create_agent()` must do no slow work (no network, no model warm-up). Runtime must already be running before 10:00. |
| Registration | Admin creates one tournament per game. You press **Register my agent** on the dashboard's Tournaments page. Button is enabled only when: agent is Self Hosted, has an active Official Agent Key, and your **event registration is complete** (the readiness codes mention a questionnaire, Rules, and a *repository*). | Registration is per tournament, so we register for the Pokémon one only. Your fork URL is probably the "repository" they want: keep it secret-free. |
| Runtime flags | **Resolved Oct 6 evening: upstream PR #5 merged and is now in our branch.** `python -m agent --match` plays your Testing seats, `--tournament` plays tournament games, both flags together play both in one process. Claim codes and the old `ALTRUAGENT_API_KEY` are retired; the only platform credential is `ALTRUAGENT_OFFICIAL_AGENT_KEY`. | `.env` needs just the Official Agent Key (plus our Anthropic variables). Our agent code was untouched by the merge. |
| Python | Repo needs 3.11+. Your laptop has **Python 3.14.4**. | **Resolved Oct 6:** `.venv` created, `pip install -e ".[dev]" anthropic` succeeded, and the starter's 498 tests pass on 3.14. No need for 3.12. |
| Claude API (Opus 5.5, Sonnet 5.5) | Depth via `output_config.effort`; structured JSON via `output_config.format` (`json_schema`); a reply can stop with `stop_reason: "refusal"` or `"max_tokens"`. | Provider sends effort `low` for battle turns and `high` for Team Preview, no `thinking` parameter, checks `stop_reason`, and uses 18 s / 8 s timeout ceilings with `max_retries=0` so we own retries and fallback. |

---

## 3. Pokémon vs Werewolf (closed: you chose Pokémon)

Short record of why that is also the right call:
- Pokémon's draft and battle have **perfect information about builds** and well-defined math, so code can carry most of the weight and the LLM adds judgment on top. Werewolf's edge was almost entirely prompting, under a 120 s shared chat window with 7 agents.
- Pokémon's battle clock (about 50 s per decision as the server reports it, 420 s bank) still leaves room for a low-effort model call (median ~3 s, p95 ~8 s). The 15 s draft is pure code.
- Pokémon has the bigger, deeper bracket (6 Swiss rounds + top 32 best-of-3), so a consistent agent gets rewarded.

---

## 4. Architecture: hybrid, code first, LLM on top

As built (the Oct 6 sketch had only `memory`, `log`, `data`, `draft`, `lineup`, `battle` and `prompts` under `pokemon/`):

```
agent/
  agent.py                # create_agent() -> PokemonAgent: routing, clock budget, model call, row gate, fallbacks, log
  llm/
    anthropic_provider.py # AnthropicProvider (official SDK) + FallbackProvider chain (Opus 5.5 -> Sonnet 5.5)
  pokemon/
    memory.py             # MatchMemory: both sides' full card sets, the battle log read every turn, restart rebuild
    log.py                # DecisionLog: one JSON line per decision -> logs/<session_id>.jsonl
    data.py               # type chart, stat formula, species/move/catalog lookups from data/*.json
    draft.py              # PURE CODE: the counter-aware draft search (and the older greedy scorer as its fallback)
    matchup.py            # 1-on-1 races between cards, team model (data/draft_model.json), lead openings
    counterpick.py        # a counter-picking drafter's scorer, for the draft search's opponent replies
    lineup.py             # Team Preview: four-against-four game, leads, lead veto, the table for the model
    preview_sim.py        # the simulated openings behind the deep Team Preview
    battle.py             # the turn sheet: speed, damage, threats, warnings, switch-ins, scored rows
    opponent.py           # the opponent class's per-slot prediction
    replies.py            # opponent priors (its reply mixture is off by default)
    match_state.py        # per-match cache reset
    prompts.py            # system prompt and its variants, Team Preview instructions
    tuning.py             # every weight and switch
  versions/, arena.py     # development: alternative players and the seat-based arena
data/                     # dex, moves, card catalog, fitted team model, per-card and first-pick tables, tiers
logs/                     # gitignored; decision logs and runtime logs
scripts/                  # run_tournament.sh (the launcher) and development tools
sim/                      # development: the local Showdown simulator
tests/                    # pytest, no network
```

**Reuse, don't rewrite** (all already in the repo):
- `examples/llm/pokemon.py` — `lineup_choice` and `doubles_choice` already turn the server's template into a bounded choice and **validate** the model's answer (`build()` rejects wrong species, bad targets, double-switch into one Pokémon, double pass). We import these and route every lineup/turn answer through their `build()`.
- `examples/llm/base.py` — `Choice`, `InvalidChoice`, `object_schema` (strict JSON schema builder).
- `examples/smoke_agent.choose_action` — a deterministic, always-legal move for any Pokémon phase. This is the **last-resort fallback** everywhere.
- `examples/llm_agent.py` — the `_ask` retry-once-with-the-error pattern and the `LLMProvider` protocol (`complete_structured(messages, schema_name, schema) -> dict`). We implement that protocol, so the rest of the example machinery works unchanged.
- `tests/test_llm_agent.py` — `FakeProvider` and the Pokémon state fixtures; copy their shape for our tests.

### 4.1 The decision flow per phase

**Draft (code only, must return in well under 1 s)**
As built and on by default (`draft_v2`): for every offered card the search plays the rest of the snake draft out,
trying at the opponent's next pick the team model's three best replies and a counter-picker's three best, and values
each finished draft as the four we would bring against their three likeliest fours under the team model fitted on
simulator games; a first-pick policy adds to our first two picks, and picks within 1 % win chance are drawn by a
secret per-match seed. The Oct 6 scorer below is the fallback if the search raises. As planned on Oct 6:
Score every offered card with `score_pool(card, my_roster, opp_roster, picks_left)`:
- base stat total and speed tier (from `data/pokedex.json`, falling back to the card if present);
- **type coverage**: reward offensive types we lack and defensive resistances to what the opponent has already drafted; penalize stacking a shared weakness;
- role balance: physical vs special attackers, at least one speed-control/support set (Fake Out, Tailwind, Trick Room, Icy Wind, redirection) by pick 4–5;
- **denial**: small bonus for taking a card that best completes the opponent's roster when our own top two scores are close;
- a deterministic tiebreak so the same pool always drafts the same way (reproducible tests).
Store the **full card** of every pick, ours and theirs, in `MatchMemory`.

**Team Preview (code shortlist, LLM chooses, 85 s as the server reports it)**
As built: code plays our 15 fours against their 15 as a small game under the team model, picks leads with a two-turn
opening model and a lead veto, then spends about 20 s simulating the first turns of the best plans
(`preview_sim.py`), and keeps the top 3. The model (Opus 5.5, effort `high`) sees the 3 plans with their numbers, a
matchup table and the opponent's exact sets, and chooses the lineup. Answer goes through `lineup_choice.build()`.
Timeout or error -> code's #1.

**Battle turn (code computes, LLM judges, code gates; about 50 s per decision from a 420 s bank)**
Code builds a compact **turn sheet** every turn:
- speed order of the 4 active Pokémon (base stats + known EVs/nature, boosts, Tailwind/Trick Room/paralysis);
- for each of our legal moves vs each target: type multiplier, STAB, rough damage as a % of the target's current HP (Gen 9 formula, level 50, using the opponent's known EVs/nature/item from the draft);
- the opponent's **known** moves per active Pokémon (from their drafted cards), with their rough damage to us, and flags like "can OHKO slot 0", "has Protect", "has Fake Out", "priority move";
- a ranked list of scored candidate full turns (both slots), e.g. "double into opp A", "Protect slot 0 + attack", "switch slot 1 to resist": as built, the best four plus up to two Protect / switch / redirection rows that cover a slot in a KO range.
As built, the model (Opus 5.5, effort `low`) sees the sheet plus the per-slot options from the server and answers with a row number (or a proposed joint action, the override), its opponent prediction, a win condition and a public `reasoning_summary`. The row gate and the lethal recheck apply (see *Final state*), and `doubles_choice.build()` validates the payload. Any failure -> retry once with the error -> row 0 -> `smoke_agent` move on an exception.

### 4.2 Per-match memory and decision log
- `MatchMemory` (one per process, created in `create_agent()`): `my_cards`, `opp_cards` (full sets by species), `first_drafter`, `my_lineup`, `opp_lineup_seen`, per-turn `turns[]` (what we played, what the opponent did, HP after), `fallbacks`, latencies.
- `DecisionLog` appends one JSON line per decision to `logs/<session_id>.jsonl`: phase, turn, compact state summary, the turn sheet, candidates, the model's answer, which model answered (Opus / Sonnet / none = code), latency ms, validation errors, and the final payload. **Never** any key, token, or env value. This is what we read after every test match to answer "why did it do that".

### 4.3 The Anthropic provider and fallback chain
`AnthropicProvider.complete_structured(messages, schema_name, schema)`:
- `anthropic.Anthropic(timeout=<per-provider>, max_retries=0)`; the SDK reads `ANTHROPIC_API_KEY` from the environment after `load_dotenv()`. The key is never stored on our object, never in `repr`, never logged. James's key is a personal key spanning several workspaces, so every request also carries the `anthropic-workspace-id` header, read from `ANTHROPIC_WORKSPACE_ID` in `.env` (verified live Oct 6).
- Request: `model`, `max_tokens` 4,000, `system` (stable, marked `cache_control` ephemeral so repeated turns get the cache discount; a second cached block carries the match's known sets), `messages=[{"role":"user", ...}]`, `output_config={"format": {"type": "json_schema", "schema": schema}, "effort": ...}` (`low` for battle turns, `high` for Team Preview). No `thinking` parameter.
- Check `stop_reason`: `"refusal"` or `"max_tokens"` -> `ProviderError`. Parse the first text block as JSON; non-object -> `ProviderError`.
- Errors are mapped most-specific-first (`RateLimitError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError`) to `ProviderError` with the HTTP status and the API's own error type and message (trimmed, anything key-shaped removed), never the request. Every call also runs under a hard wall-clock guard (its timeout + 2 s).
- Server-side refusal fallback (`fallbacks: "default"` with the beta header) is available but **not** used: our own chain already covers refusals and keeps one code path. Can be added later if logs show refusals.

`FallbackProvider([Opus 5.5 @ 18 s, Sonnet 5.5 @ 8 s])` tries each in order within one deadline for the whole decision and raises only if all fail; the caller then uses code. The model ids default to `claude-opus-5-5` and `claude-sonnet-5-5` in code; `.env` can override them (`AGENT_MODEL`, `AGENT_FALLBACK_MODEL`) so cheap test runs can use another model without code changes.

### 4.4 Rough cost per match
(The Oct 6 estimate, written when Fable 5.1 was the planned judge. The agent that competes runs Opus 5.5; measured on
Opus: about $1.50 for a 26-decision self-play match before caching (§5j) and about $0.36 a game against the sparring
partner (§5n).)
Prices: Fable 5.1 $10 in / $50 out per million tokens; Sonnet 5.5 $2 / $10. A battle decision sends roughly 4–6K input tokens (system prompt cached after the first turn) and produces roughly 1–2K output tokens including low-effort thinking.

| Item | Calls per match | Fable 5.1 | Sonnet 5.5 |
|---|---|---|---|
| Draft | 0 (code) | $0 | $0 |
| Team Preview | 1 | ~$0.12 | ~$0.03 |
| Battle turns | ~12–20 | ~$0.10–0.15 each | ~$0.02–0.03 each |
| **Per match** | | **~$2–3** | **~$0.40–0.60** |
| Tournament day (6–11 games) | | **~$15–35** | |
| Testing (40–60 matches) | | ~$100–150 on Fable, ~$25 on Sonnet | |

Budget approved: **$250**. Default to Fable 5.1 for testing so what we tune is what we play; drop to Sonnet 5.5 for bulk self-play if spend passes ~$150 before validation week. (Superseded: Opus 5.5 became the judge on Oct 6 night, §5d.)

**Measured Oct 6 (offline dry run, real Fable 5.1, effort low):** Team Preview call 2.3K in / 130 out, 7.1 s. Battle turn call 4.6K in / 100 out, 4.0 s. At $10/$50 per MTok that is about $0.03 per lineup and $0.05 per turn, so **roughly $1 per match on Fable**, well under the table above. The system prompt (~1K tokens) is written to the prompt cache; whether reads hit on later turns gets checked in the first live match.

---

## 5. Testing loop

1. **Fixtures first.** The very first test match runs with raw-state capture on, so `tests/fixtures/` gets real draft, team-preview and battle payloads. Unit tests for draft scoring, damage math and payload validity run against those in under a second, with no network and no key (`pytest`).
2. **Dashboard test matches.** Testing page -> create a Pokémon match -> both seats "Mine (self-hosted)" for self-play, or one seat Mine + leave the other open so another contestant joins (free scouting of real opponents). Then run `./scripts/run_match.sh` once; the runtime plays every seat we hold, one process per game, and the script tees the output to `logs/runtime-match-*.log` so `scripts/tally.py` can read the scores.
3. **Comparing versions.** Keep the champion in `agent/agent.py` and challengers as `agent/versions/v2.py` etc. `agent/arena.py` picks a version per seat from `AGENT_SEAT0`/`AGENT_SEAT1`, so one `--match` runtime can play v1 vs v2 against itself: `AGENT_SEAT0=agent.agent AGENT_SEAT1=agent.versions.v2 ./scripts/run_match.sh --agent agent.arena`. Swap the seats between matches. Every decision log line carries the version; `scripts/tally.py` prints wins, latency p50/p95, fallbacks, model mix and cost per version. Promotion rule: a challenger replaces the champion only after winning at least 6 of 10 self-play matches *and* showing zero fallbacks-to-code caused by bugs.
4. **Failure drills** (Day 5): kill the process mid-battle and restart it (the runtime resumes the seat); run with a bogus `AGENT_MODEL` to prove the Sonnet and code fallbacks engage; run with Wi-Fi off for 60 s.

---

## 5a. Offline strength work (added Oct 6, while waiting for the sign-up link)

- **Scenario eval suite** (`scripts/scenarios.py`): nine hand-built doubles situations with a known good play (Fake Out turn 1, Protect vs a faster KO, no Earthquake into an ally, finish the low-HP target, switch when walled, spread vs two weak targets, Trick Room speed, respect priority Sucker Punch, attack under Tailwind). `--code-only` shows what the computed fallback does; the default runs the real model chain. Every run appends to `logs/scenarios.jsonl`. **Oct 6 results: code-only 9/9; Fable 5.1 8/9** (missed the priority Sucker Punch KO; the sheet now carries explicit LETHAL warnings and the prompt a priority rule).
- **Damage math verified** against Smogon's official calculator (`@smogon/calc`, 376 generated cases in `tests/fixtures/smogon_calc.json`, `tests/test_damage_vs_smogon.py`): median error 0.5% of max HP, p90 1.4%, max 3.9%. Found and fixed on the way: multi-hit moves, always-crit moves, type-boost items, Expert Belt, Knock Off's item bonus.
- **Candidate ranking is now scored**, not fixed-order: survival factor (a slot that dies to a faster KO contributes nothing), status-move values (Spore, Rage Powder, Tailwind, Trick Room, Will-O-Wisp…), Fake Out setup on a fresh switch-in, ally damage charged against spread moves, speed-aware Protect.
- **Payload shapes** aligned with the platform guide's examples: seats keyed by agent id, `fields`/`side_conditions` for Trick Room and Tailwind, `active_pokemon` lists.
- **Effort sweep (Oct 6, three parallel runs of the 9 scenarios):** low 9/9, median ~3 s, max 16 s; medium 9/9, max 22 s; high 9/9, max 33 s (one call fell through to Sonnet 5.5, which also answered correctly). Decision: **stay on `low`**. It is the fastest and cheapest and lost nothing; the 300 s clock leaves huge margin either way. Revisit only if live matches show judgment errors the sheet can't fix. (Final state: battle turns stay `low`; Team Preview, one call per match, asks for `high`. The clock was later corrected, §5h.)
- Still to do offline: harder scenarios as live matches reveal mistakes (the current nine no longer discriminate between effort levels).

## 5b. Local self-play on the real engine (added Oct 6 night)

`sim/` runs full games on a local Pokémon Showdown engine (`npm install` in `sim/`, Node 20): an 18-card snake draft from
`sim/cards.json` (41 realistic VGC sets), Team Preview, then a `gen9vgc2025regi` doubles battle. `sim/translate.py` builds the
exact observation/template shapes the platform sends, so the agent runs its live code path; `sim/harness.py` plays N games
between any two agent specs with seat swapping and writes decision logs to `logs/sim/<spec>/` and results to `logs/sim/results.jsonl`.

    python sim/harness.py --games 200 --p1 code --p2 random        # ~10 s, free
    python sim/harness.py --games 10 --p1 fable --p2 code           # ~1 min and ~$1 per game

**Oct 6 results:**

| Matchup | Games | Result |
|---|---|---|
| code brain vs random | 60 | 88% |
| code brain vs smoke (first legal option) | 60 | 78% |
| code brain mirror | 200 | 54/46 (no first-drafter or seat bias: 48% / 46%) |
| **Fable 5.1 (full agent) vs code brain** | 44 | **82% (36-8)**; the 20-game series after the lineup fix went 16-4; the 10-game series against the *tuned* code brain, with the hard deadline in place, went **10-0** (no fallbacks, no slow decisions, ~48 s per game) |
| code brain vs random-draft ablation | 200 | 68% (the draft scorer is worth ~18 points) |
| code brain vs naive-battle ablation | 200 | 70% (the turn sheet + ranking is worth ~20 points) |

Fixes found by the simulator: the "both fainted, one reserve" case (one slot must `pass`), smart switch targets, a Fable turn
that hit the 2,000-token output cap (now 4,000). Every engine choice our agent produced was accepted (0 rejected choices).

Use it for: heuristic tuning by win rate (hundreds of free games per experiment), Fable-vs-code validation, and reviewing
Fable's losses turn by turn against the code's top candidate.

## 5c. Heuristic tuning by self-play (Oct 6 night)

`agent/pokemon/tuning.py` holds ~30 weights with defaults; each agent carries its own copy, and `sim/sweep.py` plays
variants against the baseline. First sweep: every knob at 0.5x and 1.5x, 300 games each (58 variants, 17,400 games,
~15 min). **Result: everything landed within ±6 points of 50%**, i.e. at the noise floor for 300 games, so the current
defaults are not badly mis-set anywhere. Four knobs moved consistently in one direction at both ends (lower
`draft_support_cap`, higher `switch_weak_bonus`, lower `ally_damage_w`, higher `fakeout_base`). **Confirmed at 1,000 games: the four together beat the old defaults 55.7% ± 3.1, so they are now the defaults** (annotated in `tuning.py`). Individually only `draft_support_cap=15` cleared the bar (53.9%). The opponent-lineup prediction flag measured 49.8%: a null result, kept off. Rule: adopt a change only when its 95% interval clears 50% at 1,000 games.

**Hardening from the simulator (Oct 6 night):** a Fable call once hung for 18+ minutes with the socket open despite the SDK's 40 s timeout. Every model call now runs under a hard wall-clock deadline (timeout + 5 s; later + 2 s, §5h) in a worker thread; past it, the agent treats the call as failed and falls back. The 300 s turn clock can no longer be eaten by a stuck connection.

## 5d. Which model should judge? (Oct 6 night, local self-play)

Same agent, same code brain, same prompts; only the judging model differs (`agent/versions/model_variant.py`). Ten games each,
seats alternating:

| Judge vs Fable 5.1 | Result | Latency p50 / p95 | Price |
|---|---|---|---|
| Opus 5.5 | **7-3** | 3.4 s / 8.4 s | $4 / $20 per MTok |
| Sonnet 5.5 | 5-5 | 1.9 s / 5.5 s | $2 / $10 per MTok |
| (Fable 5.1 itself) | — | 4.3 s / 20 s | $10 / $50 per MTok |

**30-game series: Opus 17-13.** Combined with the first ten, **Opus 24-16 over 40 games (60%, roughly ±15, so not statistically separated from even)** against Fable judging the same agent.
Hard positions (6 positions × 3 repeats, acceptance rules reviewed by hand): Fable low 12/18, Fable high 12/18, Opus 12/18, Sonnet 12/18, and
all four made the identical choice on all 18 runs. On model-answered turns each judge takes the code's top candidate ~46–48% of the
time with the same override rate in won and lost games. Fable had 0 real fallbacks; one 40 s timeout was rescued by Sonnet.

**Decision (Oct 6 night, James): Opus 5.5 is the judge.** Not weaker in any measurement, ahead in games, half the p95 latency.
`AGENT_MODEL=claude-opus-5-5` in `.env` and `DEFAULT_MODEL` in `agent/llm/anthropic_provider.py`; Sonnet 5.5 remains the fast fallback;
Fable 5.1 is one `.env` line away. Re-check after the first live test matches.

## 5e. Game review: last five Opus-judged games (Oct 6 night), ratings out of 10

Reviewed turn by turn against the code's top candidate (3 wins, 2 losses).

| Area | Rating | Evidence |
|---|---|---|
| Team building (draft + lineup) | **5.5** | Draft takes strong individuals (Garchomp, Iron Bundle, Calyrex-Ice) but builds little synergy: the two losses had all-offense rosters with no speed control or redirection used. Lineup benched the Ground type (Electric-immune) against a Specs Miraidon that then swept; lead Sneasler into a Scarf Urshifu whose multi-hit breaks the sash. |
| Move choices | **7** | Mostly sound: Protect vs lethal, priority sweeps, focus fire, Trick Room timing. Two costly overrides of CORRECT warnings: believed a sash survives Surging Strikes; dismissed a lethal range as "overstated" because of -4 SpA when the number already included it. Both fixed (prompt + sheet) and verified on scenarios. |
| Timing / tempo | **6.5** | Good: Trick Room set turn 1 and ridden to a win; Fake Out and Thunderclap used for tempo. Weak: turn-1 lead matchups lost a Pokémon immediately in both losses. |
| Strategy overall | **6** | Coherent when a mode exists (TR, sun); otherwise turn-by-turn with no win-condition planning, no reading of the opponent's Protect cycle, no sacrifice/positioning concept. |

**Fixes landed from the review (Oct 6 night):** prompt rules (numbers already include modifiers; sash fails vs multi-hit; Unseen Fist pierces
Protect; redirection), sash status notes and multi-hit flags in the sheet, probability-weighted survival and Protect scoring, Unseen Fist handling,
a redirection candidate, three new scenarios. Opus on the hard set: 14/16 after the fixes (was 12/18 equivalent).

**Status of the five (Oct 7, early):** 1 and 2 implemented behind tuning knobs (`lineup_answer_w`, `lead_priority_multihit_w`;
`draft_mode_w`, `draft_role_w`, `draft_deny_mode_w`) and being measured by 1,000-game sweeps before they become defaults. 3 implemented:
our own Protect last turn is tracked (a repeat is discounted to 1/3) and their Protect is inferred when a targeted Pokémon's HP did not move.
4 implemented: the answer schema has a `win_condition` field and the prompt asks for a two-turn plan; Opus 7/8 on the hard set with it.
5 pending the two 30-game series (Fable vs code, Opus vs code, same seed) whose losses will seed new hand-checked scenarios.

**Item 1 sweep result (1,000 games per variant): no gain.** `lead_priority_multihit_w` 2/4 → 49.3% / 50.2%; `lineup_answer_w` 4/8 → 46.5% / 48.2%; combined → 48.3%. The defensive-answer rule costs more offense than it saves in self-play, so both knobs stay at 0 (code kept for live-match re-test). A good reminder that a convincing single-game story (Garchomp vs Miraidon) is not a measured gain.

**Item 2 sweep result (1,000 games per variant):** `draft_mode_w` 10 → 52.6% ± 3.1 (borderline, being re-tested with a fresh seed), 20 → 49.3%; `draft_deny_mode_w` 5 → 50.8%; `draft_role_w` 5 → 49.3%, 10 → 45.4% (hurts: forcing roles costs raw strength); combined → 51.1%. Only a mild mode preference shows any sign of value; role quotas and denial do not. Re-test of `draft_mode_w` 10 with a fresh seed: 51.7% ± 3.1; pooled over 2,000 games 52.2% ± 2.2, whose lower bound sits at 50.0. **Not adopted** under the rule (interval must clear 50%); kept at 0 as a candidate to re-test against real opponents.

**Where to improve next, in order of expected impact:**
1. **Lineup: defensive answers.** Require at least one brought Pokémon that resists or is immune to each of their two strongest attackers' main STAB; weight leads against their likely leads' multi-hit and priority. (Would have brought Garchomp vs Miraidon.)
2. **Draft: build around a mode.** Once a Trick Room / Tailwind / weather setter is picked, value its partners; reserve roles (speed control, Fake Out/redirect, a defensive answer to their best attacker); deny their mode-completing piece.
3. **Opponent modelling across turns.** Track their Protect usage per Pokémon and their revealed patterns; the sheet is currently stateless turn to turn.
4. **Win-condition framing in the prompt.** Ask the model to name the win condition (which of theirs must die, which of ours must live) and plan two turns ahead.
5. **More hard scenarios from real losses**, verified by hand before they are trusted (two of today's were wrong on first writing).

## 5f. Fable vs Opus, same rubric, same pools (Oct 7 early)

Two 30-game series vs the tuned code brain with the same seed (identical draft pools, and the draft itself is code, so both
judges drafted the same six). **Credits ran out mid-run**: from about game 22 on, every model call failed with
"credit balance is too low" and the code brain played those turns (the fallback chain worked; the agent never crashed; the
provider now prints a loud one-time CREDITS EXHAUSTED warning). Only games with every decision model-played count:

| Judge | Clean games | Record |
|---|---|---|
| Fable 5.1 | 15 | 11-4 (73%) |
| Opus 5.5 | 18 | 16-2 (89%) |

Rubric on the last five clean games of each (Fable 5-0, Opus 4-1):

| Area | Fable 5.1 | Opus 5.5 | Notes |
|---|---|---|---|
| Team building | 6 | 6 | Draft is code and identical. Both chose sensible lineups; Opus's one loss brought Garchomp over the code's Maushold and lost the lead on turn 1. |
| Move choices | 8 | 7 | Fable: sash-break sequencing ("Electro Drift breaks the sash, Body Press finishes"), priority Thunderclap timing, Trick Room order. Opus: Follow Me to soak Fake Out, good Protect calls, but in its loss Protected three turns in a row (now discounted by item 3). |
| Timing / tempo | 7.5 | 7 | Both set their mode on turn 1 when available; Fable sequenced priority and speed modes slightly better. |
| Strategy | 7 | 6.5 | Fable's lineups named a second speed mode and used it; Opus planned one turn at a time more often. |

**Rerun after credits were added (seed 501, 30 games each, zero fallbacks, drafts identical in all 30 games): Opus 21-9, Fable 20-10.** The two judges differed on only 7 of 30 games (4 Opus-only wins, 3 Fable-only wins): a statistical tie.

Rubric on the last five games of the clean seed-501 series (identical drafts; lineups identical in 4 of 5; Fable 3-2, Opus 4-1):

| Area | Fable 5.1 | Opus 5.5 | Notes |
|---|---|---|---|
| Team building | 6.5 | 6.5 | Draft is code; lineups near-identical. Both lost game 26 with the code's top lineup into an Iron Hands / Maushold / Farigiraf Trick Room team: a lineup-scorer blind spot (Population Bomb multi-hit, Follow Me), not a judge difference. |
| Move choices | 7.5 | 7.5 | Both: sash-break sequencing, Protect reads, Armor Tail awareness, Wide Guard vs Earthquake. Both locked Miraidon into Electro Drift with a Ground type still standing (game 25); Fable overcommitted Flutter Mane into Lunala's lethal range after naming it as the piece to keep (game 29, lost). |
| Timing / tempo | 7.5 | 7 | Fable timed Tailwind and priority slightly better; Opus's game 25 dragged to 11 turns on the Choice lock. |
| Strategy | 7 | 7.5 | Opus's game-29 sequencing (Iron Bundle first, Flutter Mane as the closer) preserved resources; Fable's win_condition statements were explicit but not always obeyed. |

Overall ~7.1 each: the judges are interchangeable at this level. The differences that decided games were shared code-side gaps.

Reading: on this rubric Fable's play reads a little richer; on results Opus is ahead in every head-to-head and vs-code series run
tonight (24-16 vs Fable directly; 16-2 vs 11-4 on identical pools). Neither gap is statistically clean. **Decision unchanged: Opus
judges**, with the explicit plan to re-run this comparison on the live server once credits and the dashboard are available.

## 5g. What game 26 really was: the damage model was blind to abilities (Oct 7)

Replaying the game both judges lost (Hatterene + Incineroar into Iron Hands + Maushold) against Smogon's calculator showed
the turn sheet's numbers were wrong in a way the judges could not see: Maushold's Population Bomb was computed without
**Technician** (a third too low), so the sheet never warned that it one-shots Hatterene at full HP. Twice I hand-checked a
number, called the code wrong, and was wrong myself (forgot STAB, then forgot Technician). Rule from here: **damage is never
hand-checked; it is checked against @smogon/calc** (now a dev dependency of `sim/`).

Audit of the 42-card pool found everything the old model ignored, all knowable from the drafted sets:

| Missing before | Cards it mattered for |
|---|---|
| Technician | Maushold (Population Bomb 20 → 30 per hit, ×10) |
| Sword / Beads / Vessel of Ruin (−25% to everyone else's Def / SpD / SpA while on the field) | Chien-Pao, Chi-Yu, Ting-Lu |
| Guts + burn (Atk ×1.5, no burn penalty), Facade ×2 when statused | Ursaluna (Flame Orb) |
| Booster Energy / Protosynthesis / Quark Drive (highest stat ×1.3, speed ×1.5) | Iron Bundle, Raging Bolt, Flutter Mane in sun, Iron Hands in Electric Terrain, Gouging Fire |
| Hadron Engine / Orichalcum Pulse (SpA / Atk ×1.33 in their terrain / weather) | Miraidon, Koraidon |
| Terrain (×1.3 same-type grounded moves; Earthquake halved in Grassy; Expanding Force 120 BP spread in Psychic; priority blocked in Psychic) | Rillaboom, Indeedee-F, Miraidon, Hatterene |
| Reflect / Light Screen (×2/3 in doubles) | Grimmsnarl |
| Stamina on multi-hit (every hit lands on +1 more Def) | Archaludon |
| Body Press (uses Def), Foul Play (target's Atk), Weather Ball (type/power in weather), Eruption (HP-scaled), Heavy Slam (weight), Mind's Eye (hits Ghosts), Meteor Beam / Electro Shot (+1 SpA before the hit), Sacred Sword (ignores stages), Knock Off vs unremovable items / popped seeds, Ruination (half HP) | Zamazenta, Archaludon, Farigiraf, Pelipper, Torkoal, Iron Hands, Ursaluna-Bloodmoon, Glimmora, Lunala, Chien-Pao, Ting-Lu |

All of it is in `agent/pokemon/battle.py` now (`FieldState`, `effective_move`, `damage_percent(field=...)`), read from the
observation by `field_from_obs`. Ground truth: `scripts/gen_smogon_fixture.js` writes 1,314 calculator cases with abilities and
field effects on; `tests/test_damage_vs_smogon_full.py` requires median error < 1% of max HP, 90th percentile < 2.5%, worst < 6%
and one close case per mechanic. Result: **median 0.3%, p90 0.9%, worst 4.0%.**

Two more things the replay showed, both fixed:
- **Stale Fake Out.** Both judges picked Fake Out on a Pokémon's third turn out (it fails), once into Armor Tail as well. The sheet
  now lists such options as `FAILS` / `BLOCKED` with no targets, the opponent's Fake Out note says `FRESH` only when it is live, and
  a priming field in the scenario runner makes turn > 1 scenarios honest about who just switched in. New basic scenario
  `stale_fake_out`.
- **Overkill.** The code's top candidate doubled into a 30% Gyarados that Rock Slide already KOs, instead of finishing the sash
  Whimsicott (the one hard scenario that had been failing code-only). New candidate `finish_the_other`: when one slot's attack
  guarantees a KO, the other slot takes the remaining foe, and if the first slot's spread hit breaks that foe's sash first, the
  second hit is recomputed against the HP that will actually be left.

Also added: a real-damage lead check at Team Preview (`lead_ohko_w`, which opposing set OHKOs a lead before it moves, Intimidate
applied), a knob around the Choice-lock re-ranking so the sweep can A/B it, and the prompt now says FAILS/BLOCKED options are
off-limits and that the numbers include abilities, terrain and screens.

After the change: tests 675 green; scenarios **18/18 code-only** (was 17/18); code vs random 94%, vs smoke 85% (93/82 before).
Sweeps (1,000 games each vs the tuned defaults, seed 11): `lead_ohko_w=3` 50.3% ± 3.1, `lead_ohko_w=6` 50.6% ± 3.1,
`choice_lock_rerank=0` 50.1% ± 3.1. None clears the adoption bar, so the lead check stays off (an Intimidate lead makes true
turn-1 OHKOs rare in this pool) and the Choice-lock re-ranking stays on (it is logically right, free, and only matters with a
Choice holder on the field).

Opus on all 18 scenarios after the change: 15/18 on the first pass. Two of the three "failures" were acceptance rules written
under the old, too-low numbers: with Sword of Ruin modelled, Kingambit's Sucker Punch and Chien-Pao's Icicle Crash are both lethal
to Flutter Mane in `break_sash_then_ko`, so pivoting it out is right, and in `sucker_punch_respect` switching the Sucker Punch target
out (the move fails against a switching target) is as good as Protect. Both rules were broadened. The third, `stale_fake_out`, was
a real lesson: the sheet called an 85-101% hit (a 4% KO chance) LETHAL and Opus burned a second Protect in a row on it. Warnings are
now graded: LETHAL only when guaranteed or at least a coin flip, otherwise RISK with the KO chance spelled out, and every
"possible" threat row carries its `ko_chance_pct`.

Rerunning the two hardest ones three times each still showed the model attacking with a Pokémon the sheet had marked LETHAL
(a priority Sucker Punch it had read as "it moves first under Tailwind", and a speed tie it read as a win). So the agent now
does a **lethal recheck**: if a valid answer leaves a LETHAL-flagged slot attacking, the model gets that exact warning quoted back
and answers once more; its second answer stands either way (trades can be right), and the decision log records `recheck`.
(Final state, `recheck_v2`: it fires only for a LETHAL or EXPOSED slot that row 0 Protects or switches, and points the
model back to row 0.) With
it: `stale_fake_out` 2/2, `sucker_punch_respect` 3/3, `break_sash_then_ko` 3/3 under the broadened rule (the recheck fired on
3 of the 6 and flipped the answer each time). Scenario suite now 18 (10 basic + 8 hard); code-only 18/18.

**Seed-501 rerun on the v2 damage model (Oct 7, before the draft-memory and warning fixes): Opus 18-12 (60%)**, 266 model
decisions, zero fallbacks, 26 rechecks, latency median 3.1 s / p95 7.7 s. Down from 21-9 on the old numbers, but the code
opponent received the same corrected numbers, so this measures the judge's edge over the code brain, not the fix; 30 games is
also only ±17 points. Seven of the twelve losses had a lead fainting on turn 1 to a faster attacker the sheet either flagged as
"RISK" at 47% or (game 17) failed to flag at all; both are fixed below. The second rerun (all fixes in) **ran out of Anthropic credits after 33 model decisions**: the remaining 205 decisions were
code-only fallbacks (the chain worked, no crash, CREDITS EXHAUSTED warning printed 26 times) and the 15-15 result is just code vs
code. It needs credits added and a rerun before it counts.

**Human seat (Oct 7).** `python sim/harness.py --games 1 --p1 human --p2 fable --verbose` lets James play the agent from the
terminal: numbered draft picks with full sets, Team Preview, and per-turn options with targets; `h` on a battle turn prints the
code brain's warnings and top candidates. Building it exposed a real bug: **the agent's own last draft pick never reached
memory as a full card.** The server only talks to us on our own turns, so there is no draft observation after our final pick,
and Team Preview only sends species/types/base stats. In every simulated game so far one of our six went into the lineup
scorer blank (no moves, no item, no EVs) and, if brought, into the battle sheet with wrong stats. Fixed in two places: the agent
records the card it just took (`MatchMemory.record_pick`), and Team Preview recovers any roster member it never saw as a
card from the pool. Regression test added. Code brain after the fix, 300 games each: **97% vs random (was 93-94), 85% vs smoke** (unchanged). Note
the seed-501 rerun below started before this fix and does not include it.

**Warning bug found while reviewing the seed-501 losses (Oct 7).** The LETHAL check decided "we act first" using the priority of
whatever move the code itself ranked best. In game 17 the code liked Follow Me (+2), so no warning was printed for Koraidon's
Flare Blitz into Maushold; the model picked Population Bomb instead and lost Maushold on turn 1. Warnings now use the slot's
ordinary speed and add "(a priority move from this slot would go first)" when that escape exists. Risk is graded three ways:
LETHAL (guaranteed or at least 50%), HIGH RISK (20-50%: the old text called a 47% chance of losing Tornadus "usually worth playing
through"), RISK (under 20%). Threats that are slower than us but KO us after we move are now listed as EXPOSED notes, so the
model can see a trade before it makes one. Regression test added.

**Effort sweep (Oct 7, all 18 scenarios, Opus, recheck on):** low 18/18, median 3.2 s, p95 8.1 s; medium 17/18, median 7.8 s,
p95 14.3 s (it missed `intimidate_the_dancer`). The recheck fired 4 times at low and 2 at medium. More thinking bought nothing here
and doubled latency, so **effort stays low**; the recheck is doing the work the extra thinking was supposed to do. (Final
state: battle turns `low`, Team Preview `high`.)

### 5h. Oct 7 afternoon: upstream merge, the real battle clock, reliability drills, readiness

**Upstream merged** (`git merge upstream/main`, one trivial conflict in the placeholder docstring of `agent/agent.py`,
ours kept): PR #5 (`--match` plays Testing seats, `--tournament` plays tournament games, `--claim` is gone) and PR #7
(runtime resilience: retries inside a game, 40 s MCP timeouts, a hiccup never becomes a no-show). One upstream test
(`test_realtime.py::...red_alert...`) asserts the *placeholder* agent's message; skipped in the fork with a reason. 845 pass, 1 skipped.

**Clock correction.** Upstream GAMES.md now states the battle runs **Showdown's VGC timer: 90 s Team Preview, 55 s per
battle decision, 420 s total bank per player per battle**; an empty bank is a forfeit. §2 had 300 s. Changes:
`DEFAULT_TIMEOUT_SECONDS` 40 -> 18, `DEFAULT_FALLBACK_TIMEOUT_SECONDS` 12 -> 8, grace 5 -> 2 (chain worst case 30 s);
`_ask` makes a second model call (retry on an invalid answer, or the lethal recheck) only while under
`SECOND_CALL_CUTOFF_SECONDS` = 10 s have passed, else the first valid answer stands / the computed move plays (2 tests).
Worst-case decision ~40 s; measured median 3.1 s, p95 ~8 s. Bank math: 12 turns x 5 s = 60 s of a 420 s bank.
(Superseded: the cutoff was replaced by the per-decision clock budget described in *Final state*, which reads the
server's own clock: about 50 s per decision when it asks, 85 s at Team Preview.)

**Reliability drills (§5.4), all passed, no credits used:**
- A. Bogus model names (`AGENT_MODEL=claude-bogus-model AGENT_FALLBACK_MODEL=...`): both failed (the 400 was the credit
  error, which arrives before the model-name check), code move in 823 ms / 484 ms, CREDITS EXHAUSTED banner printed.
- B. Unreachable API (`ANTHROPIC_BASE_URL=http://127.0.0.1:9`): "connection failed" on both models, code move in 17 ms / 1 ms.
- C. Restart mid-battle: new `sim/harness.py` spec `code@restart3` throws the agent away at battle turn 3 and builds a fresh
  one with no draft/preview memory (what the runtime does when a game process dies). 6 games, 0 rejected choices, 3-3 vs code.
  The drill against the real runtime (kill the process, let it resume the seat) still needs the Official Agent Key.

**Readiness, honest (asked by James after $150 of credits):** engine ~8/10, tournament entry ~5/10. Done and proven: legal
play end to end in the sim, 845 tests, validators on every payload, fallback chain under real failure (twice), damage
model within 4% of @smogon/calc, draft < 50 ms, code 97% vs random / 85% vs smoke, 19/19 scenarios, Opus+code 18-12 vs code
(60%, not statistically settled). Not done, by risk: (1) never connected to the real platform; observation format, card pool
and field names are from docs + the example code; blocked on the sign-up link / Official Agent Key; (2) the 42-card pool is
our guess; (3) real-runtime restart drill; (4) cold-boot dry run of `run_tournament.sh`; (5) registration. Credit plan:
stop running 30-game Opus series (they measure, they don't improve; ~$0.15-0.20 per decision); hold ~$40 for real platform
test matches once the key arrives plus ~$15-25 for tournament day.

### 5i. Oct 7 evening: the first live match on the real platform, and what it found

**Setup.** James added the Official Agent Key and registered the agent as **Ace Trainer Claude**. `--check-tournament`
showed ✓ on all six lines once the key line was uncommented in `.env` (his paste landed on the example line). His save also
reverted two `.env` lines: `AGENT_MODEL` went back to Fable (fixed to Opus) and `ANTHROPIC_WORKSPACE_ID` vanished; the API
confirms the personal key must send that header, so **James must re-add the workspace id** (Anthropic Console → Settings →
Workspaces) before any model call works. The retired `ALTRUAGENT_API_KEY` line is ignored by the runtime.

**Match 1 (self-play, both seats ours, code-only, `AGENT_CAPTURE_DIR=tests/fixtures/live`).** One `--match` runtime played
both seats. Draft: 12 picks, 3–9 ms each. Team Preview and 11 battle turns, completed, player 1 won. No errors, no rejected
moves. The real observation parsed on turn 1: speed order with real stats, damage rows, ranked candidates. Real format notes:
`weather` and `fields` are dicts keyed `UPPER_SNAKE` (`{}` / `{"GRASSY_TERRAIN": 0}`), `clock` =
`{decision_seconds_left, bank_seconds_left}` (50 s left when we are asked, 85 s at Team Preview), unrevealed opposing
Pokémon have `item: null, moves: []` (our drafted sets fill them), the battle observation has no `phase` key (top level has it),
`target_legend` names board positions. 12 of the 12 real cards are species we know; Gyarados and Dondozo were not in `sim/cards.json`.

**Two real bugs the simulator never showed (both fixed, covered by `tests/test_live_fixtures.py`, which replays every captured state):**
1. From turn 5 both seats produced **zero candidates** and the adapter's first-option default played for seven turns
   (Fake Out every turn). Cause A: a slot whose only option is `pass` (our last Pokémon stands alone) made `add()` drop every
   candidate. Fix: `lone` slots get the pass answer and `best_attack_alone`. Cause B: a stale Fake Out (`FAILS`) had its
   targets emptied and fell into the spread branch, kept its +3 priority bonus and won "best attack"; its target 0 then failed
   validation, so the one candidate was dropped. Fix: FAILS/BLOCKED rows carry `fails: True`, produce no damage rows and
   `_best_attack` skips them.
2. `weather` as a dict was read as "no weather". Fix: both reads go through `normalize_weather`, which handles dicts.

**Also from the live payload:** the agent now reads the server clock. Under 25 s left on the decision or under 90 s in the bank,
the model is skipped and the computed move plays; the second-call cutoff shrinks with the decision clock. (Superseded by
the per-decision clock budget in *Final state*.) Capture files are now
named `<phase>-<version>-seat<n>.json` (both seats and all phases used to overwrite each other; the draft states were lost).

**Baselines after the fixes:** 860 tests, 19/19 scenarios, code vs smoke 85% (200 games), code vs random 930/1000 (93%) with the fixes vs 930/1000 for the previous commit on the same seed: no regression.

### 5j. Oct 7 night: match 2 on Opus, the real pool, and the operating plan for the week

**Match 2 (self-play, full agent on Opus 5.5, both seats ours).** 26 model decisions, **26/26 answered by Opus, 0 fallbacks,
0 zero-candidate turns**, median 2.9 s, p95 7.5 s, max 8.4 s; bank never below 395 s of 420. Two lethal rechecks fired
(turns 8 and 10). Opus deviated from the code's top candidate on about a third of turns (e.g. Fake Out + Tailwind over
focus fire on turn 1, attacking instead of switching a weak slot). Tokens: 272K in / 7K out, about $1.50 at list price
before caching. Played to completion, player 1 won.

**Real pool.** `scripts/pool_from_live.py` merges every card seen in captured live drafts into `sim/cards.json`: 18 real
sets so far, 12 replaced our guesses (different EVs/moves), 6 new species (Lucario, Great Tusk, Salamence, Scream Tail,
Sableye, Dondozo). Pool 42 -> 48. Rerun it after every captured match; the simulator and scenarios then tune on real sets.

**Launcher dry run:** `./scripts/run_tournament.sh` passed pre-flight, connected, waited for tournament games, and restarted
itself after a kill. (It plays tournament games only; add `--match` for test matches.)

**What data we still need, and how many matches.** Self-play proved the pipeline (2 matches) and is now low value: the same
brain on both sides cannot lose. The signal that matters is **matches against other contestants' agents**: "Open matches" on
the Testing page, or test matches James creates with one seat left open. Target **10+ such matches before the Oct 13 freeze**,
about $1.50 each on Opus. From each: result, fallbacks (must stay 0), latency p95, bank minimum, rechecks, and every lost
battle mined with `scripts/loss_to_scenario.py` into a calc-checked scenario. Plus **one more self-play match** used for the
real-runtime restart drill (kill the runtime mid-battle, start it again, confirm it resumes the seat).

**Who runs the runtime.** James keeps one terminal open all week running `./scripts/run_tournament.sh --match` (one copy
only; it survives restarts and plays every match he creates or joins), creates/joins matches on the dashboard, and the decision
logs land in `logs/`. Claude reads the logs afterwards. Claude's background runs stop when the session ends, so they are for
supervised experiments only.

### 5k. Oct 7 night: restart drills on the real runtime (both passed), one lesson

**Drill 1, accidental and complete.** James's launcher picked up a self-play match at 17:16 and started two game processes.
He stopped it with Ctrl+C at about 17:17 and ran the launcher again at 17:17:01. The new runtime re-authenticated and
**resumed both seats** with fresh agent processes at 17:17:10 (four `config` rows in one session log). The fresh agents had
no draft or Team Preview memory and still played 44 decisions on Opus with **0 fallbacks, 0 zero-candidate turns**. Match
completed. Lesson: the gap between stop and resume was about 67 s, longer than the 55 s decision clock, so **turn 1 was
auto-played by Showdown** (no turn-1 rows in the log, the bank shrank). The launcher's automatic restart (5 s + about 5 s
to reconnect) stays inside the clock; a manual Ctrl+C and retype does not. On tournament day: never stop the launcher by hand
during a game; if the runtime dies, the launcher brings it back.

**Drill 2, forced.** At 17:17:52 a `kill -9` hit the supervisor process mid-battle. The two game processes are children with
their own connections: they kept playing and finished the match normally. The launcher only noticed the supervisor's death
once those children closed the shared log pipe (17:20:00) and restarted the runtime at 17:20:05, which connected and went back
to waiting. No game was lost. (If the game processes had died too, drill 1 shows the restarted runtime resumes them.)

### 5l. Offline training plan, Oct 7 night -> Oct 13 freeze (needs no live matches)

Live matches against other contestants remain the ground truth (James joins Open matches; the runtime plays them). Everything
below runs on this laptop and is ordered by expected value per dollar. Adoption rule for any tuning change is unchanged: the
95% interval must clear 50% at 1,000 games on the real pool, with tests, 19+ scenarios and the live-fixture replays green.

1. **Real-pool sweeps (free).** Eleven variants of the switched-off draft/lineup knobs at 1,000 games each (running Oct 7
   night, `scratchpad/sweep_realpool.txt`), then the full `--preset knobs` one-at-a-time sweep overnight. Winners -> defaults.
2. **Sparring partner (~$0.30 a game).** `agent/versions/sparring.py` = the starter's example LLM agent on Sonnet 5.5, the
   closest stand-in for a typical entrant (general model, no damage math). First 10 games code-only vs sparring (our side free),
   then 10 games Opus vs sparring. Every loss -> `scripts/loss_to_scenario.py` -> calc-checked scenario.
3. **Override grader (free).** `scripts/grade_overrides.py`: for each Opus turn that deviated from the code's top candidate,
   flag red patterns (picked a FAILS option; attacked into LETHAL with no recheck; Protect with no warning; zero-damage target
   when a damaging one existed; left a guaranteed KO on the table). Run over all Opus logs (live + sim). Patterns -> prompt
   rules or rechecks, then re-measure on the scenario suite.
4. **Scenario growth.** Real positions from `tests/fixtures/live/` and sparring losses become hard scenarios. Target 30 by freeze.
5. **Human seat.** James plays the agent (`python sim/harness.py --games 1 --p1 human --p2 code --verbose`, free; `--p2 fable`
   for Opus). A human finds blunders a mirror never will; each one becomes a scenario.
6. **Pool growth.** `python scripts/pool_from_live.py` after every captured live match; re-run baselines on the new pool.
7. **Nightly regression (free).** `pytest`, `scripts/scenarios.py --code-only --set all`, 1,000 games vs random and vs smoke on a
   fixed seed; numbers appended to this plan. Nothing ships that lowers them.
8. **Not doing:** more self-play on Opus, higher effort, bigger models (all measured; no gain).

**Results log (offline plan)**
- Oct 7 night, sweep 1 (real pool, 1,000 games each, seed 7): draft_offense_w=14 52.1 ± 3.1; draft_mode_w=8 51.3; lead_ohko_w=3 51.0;
  draft_deny_mode_w=4 50.9; draft_mode_w=4 50.5; draft_defense_w=8 50.2; lead_priority_multihit_w=3 48.6; draft_role_w=4 48.0;
  lineup_answer_w=3 47.6; draft_role_w=8 47.3; lineup_answer_w=6 47.0. **Nothing clears 50% with its interval; no change.**
  The two best and their combinations are being re-tested at 3,000 games (seed 8). draft_role_w and lineup_answer_w look
  harmful on the real pool and stay off.
- Oct 7 night, sweep 2 (real pool, 3,000 games each, seed 8): draft_offense_w=14 50.7 ± 1.8; draft_mode_w=8 50.1 ± 1.8;
  draft_offense_w=14 + lead_ohko_w=3 48.4; draft_offense_w=14 + draft_mode_w=8 48.0. **No variant clears 50%; defaults unchanged.**
  The code brain's weights are at a local optimum on the real pool; further gains must come from new logic (scenarios from real
  losses), not from re-weighting.
- Oct 7 late, first fix from the override grader: **repeated Protect**. In a self-play game Hatterene Protected on turns 5, 6 and 7
  against the same lethal threat while a 1%-HP Sneasler stood there; the code only offered the switch when Protect was unavailable.
  Now `MatchMemory.protect_streak` counts consecutive Protects by the Pokémon still in the slot, the Protect candidate's bonus
  scales by 0.33^streak, the sheet says "another one works only 1 time in 3 / 9", and `switch_threatened` is offered whenever a
  Protect was used last turn. New hard scenario `repeat_protect_pinned` (20/20 code-only). Same-seed 1,000-game baselines after
  the change: vs random 94% (was 93%), vs smoke 84% (unchanged). 900 tests.
  Gotcha: two simulator runs at once (sweep + sparring) both deadlocked on their engine bridges
  after ~15 min; run one simulator at a time.

### 5m. Oct 7 late: the project was living in iCloud, and that was the cause of every stall

`~/Desktop` is synced by iCloud Drive. Everything under `Claude Code Projects/altruagent-starter` went through Apple's file
provider. The day's sweeps wrote one decision log per simulated game until `logs/` held **87,273 files**; the sync daemon
(`fileproviderd`) ran at 90% CPU, and plain reads inside `.venv` blocked for minutes: `import anthropic, mcp, httpx` took
**more than 5 minutes** (killed), the 3,000-game sweep and the sparring run froze together, even `find logs` did not return.
Every earlier "slow import" and "stalled simulator" note in this plan has this one cause.

**Why it matters on Oct 16:** each game spawns a fresh process that imports the SDK and the agent. If iCloud is churning at
that moment the import can outlast the 4-minute connect window or the 55 s clock. Today's live matches worked only because the
folder happened to be quiet.

**Fixes.**
- The working copy is now **`~/projects/ace-trainer-claude`** (a fresh clone of the fork, outside any synced folder) with its own
  `.venv` and `sim/node_modules`. Same imports there: **1.6 s**. 898 tests pass; 50 simulated games in 2.8 s.
- On the Desktop copy, `logs/` was renamed `logs.nosync` (iCloud ignores that suffix) with a symlink at `logs`, so the runtime still
  running there keeps writing and iCloud stops syncing it. The Desktop copy is retired once the runtime moves.
- James: copy `.env` into the new folder (Claude is not permitted to touch that file), then restart the launcher from there
  between matches: `cd ~/projects/ace-trainer-claude && ./scripts/run_tournament.sh --match`.
- Rule: **never run the tournament runtime, the simulator or a sweep from an iCloud-synced folder.** Disk is 94% full (13 GB
  free); clear space before Oct 16.

### 5n. Oct 7 night: sparring results and two fixes mined from them

**Sparring, code-only vs the starter's LLM agent on Sonnet 5.5 (10 games, seed 101, clean copy): 9-1.** Games lasted 4 to 11
turns, 38 to 66 s each. The one loss (game 7) was mined:
- Draft gave them Miraidon + Lunala + Scarf Urshifu; our leads Garchomp + Salamence lost Salamence turn 1.
- Turn 2: Urshifu (Unseen Fist pierces Protect) threatened Garchomp, so the code offered `switch_threatened` into the only bench
  Pokémon, Iron Bundle, which Close Combat KOs on entry. A pivot that donates a Pokémon.
- **Fix 1:** `switch_threatened` now builds the incoming Pokémon (bench HP from the observation) and drops the switch when the
  same attacker's known moves KO it on entry, adding a warning that says so.
- **Fix 2:** the live server includes Showdown's protocol log in every observation; `MatchMemory.observe_protocol` records each
  opposing Pokémon's last move since it entered, and a Choice holder's threats list only that locked move (the simulator now
  passes its `log_tail` as `protocol_log` too). New scenario `no_suicide_pivot`.
- Earlier the same night, from the override grader: **repeated Protect** (streak counter, 0.33^streak, switch offered after a
  Protect; scenario `repeat_protect_pinned`).
State after both: 970 tests, 21/21 scenarios, simulator baselines unchanged (93-94% vs random, 84% vs smoke).

**Opus vs sparring (8 games, seed 202): 7-1.** 52 Opus decisions, 0 fallbacks, 7 rechecks, median 3.6 s, p95 11.1 s, max 13.4 s
(higher than live p95 7.5 s: the sheet has grown; still well inside the 18 s timeout). Cost about $2.90 at list for all eight
games, roughly $0.36 a game, far under the $1.50 estimate. Grader: Opus took the code's #1 on 51%, another candidate 19%, its
own move 25%; six attacks into a lethal warning, five of them after a recheck with a stated reason.
The loss (game 5) was a Scarf Urshifu (Unseen Fist) game: on turn 1 the code ranked switching Incineroar out above Fake Out on
Urshifu + Prankster Tailwind, because a guaranteed KO elsewhere cut the Fake Out base to 15 and the lethal-threat bonus was 20.
**Fix 3:** Fake Out targets the opponent that would KO one of our slots before it moves; that flinch is scored like a Protect
(+45 x KO chance) and keeps the full base, with a note that it also stops Unseen Fist. New hard scenario
`fake_out_the_scarf_urshifu`. 22/22 scenarios, 1,012 tests, same-seed baselines vs random 95% (was 94), vs smoke 85% (was 84).

**Opus pass over all 22 scenarios (Oct 7 night, after fixes 1-3): 22/22**, median 4.8 s, max 14.5 s.

**Next (in order):** to measure the judge's lift against a realistic opponent and mine its errors;
rerun `scripts/grade_overrides.py` on those logs; grow the scenario suite toward 30; keep the runtime up for Open matches
against other contestants; at freeze paste the final commit ID into the dashboard and save a redacted `.env` with the logs.

### 5o. Oct 7 late night: 0-4 against a real opponent, root-caused

Four Testing matches against another contestant, 20:50-21:16 PT, all lost; three were not close. Transcripts:
`python scripts/postmortem.py <session>` (d9405f02, 1aef9227, 7ce71025, 0a2b7c2a).

**The headline: every live battle so far was played half blind, and the simulator hid it.** The simulator's payloads were
written from the docs; the live server's differ, and each difference disabled something:

| What broke | Live payload | Effect in the losses |
|---|---|---|
| Opponent sets wiped at Team Preview | roster entries carry `moves: []`, `item: "unknown_item"`; `_enrich` merged them over the drafted cards | Lineup scored "resists 0/6 attackers" in all four matches. In battle only moves already USED were threats: Iron Hands' Wild Charge (OHKO'd Tornadus, M3 T1), Incineroar's fresh Fake Out (M3 T3), Gholdengo's Make It Rain (KO'd Baxcalibur, M4 T2) had no warning. The prompt told the model "the opponent's moves are KNOWN" and handed it empty lists. |
| Opponent's first pick unknown | when they draft first, their first card is never in our offered list | Incineroar (M2) and Dragonite (M1) played as blank 0-EV sets: Close Combat "KOs" Incineroar (did 87%). |
| Spread moves had no damage | no-target moves arrive as `targets: [0]`, not `[]` | Dazzling Gleam, Earthquake, Rock Slide, Icy Wind never ranked or shown. M3 endgame: Hatterene used resisted Mystical Fire (~15%) four times instead of Gleam (~3x); likely the game. |
| Benched Pokémon looked available | live `team` lists all six (one unbrought even `active: true`) | The model planned "Baxcalibur/Lucario can come in" (M1) and "Garchomp/Ursaluna finish it" (M3); neither was brought. Cresselia Protected four turns as our last Pokémon waiting for them. |
| Item state | revealed / "unknown_item" / null after -enditem | Unknown items overrode the card's (Assault Vest, Loaded Dice ignored); a popped Air Balloon stayed on. |

Also wrong with or without the payload: the damage model lacked Pixilate (Sylveon's Hyper Voice scored 14% into Iron Hands; it
does ~55%), Adaptability, Unaware, resist berries, Freeze-Dry and Ground vs Levitate/Air Balloon (the new live cards; checked
against @smogon/calc on the live catalog: median 0.29% of max HP, p95 1.25%). The simulator pool had 17 cards that never appear
live (Calyrex-Ice, Miraidon, Koraidon...) and 13 wrong sets, so all offline tuning ran on a pool that does not exist. Battle
logic gaps the losses exposed: no double-target threat (Landorus fell to Extreme Speed + Shadow Ball, and to Scale Shot + Draco
Meteor), no Fake Out flinch risk on our "moves first" plans, overkill valued above a KO, Protect streak reset by replacement
decisions, Helping Hand into an empty slot, a lethal recheck that argued for a pointless last-Pokémon Protect.

**The meta-cause:** our evaluation measured the wrong thing. Thousands of simulated games and 22 hand-built scenarios ran
against our own code, a random player and a weak starter agent, on a fictional pool, through a payload that never exercised
the live code paths; the live-fixture tests only asserted "no crash". Rule from now on: every live capture is replayed
through ONE agent (tests/test_live_semantics.py) and checked for what the agent KNEW; the simulator must match live payloads
field for field (parity test); offline numbers count only against opponents that play like real ones.

**Fixed (commits 9496533..4e0c121):** live catalog (data/cards.json); memory fills gaps only; catalog first picks; item
semantics; paradox boosts from the log; spread estimates; battle_roster in the prompt and observation; opponent entries filled
from cards; damage mechanics above; Fake Out flinch risk, double-target threats and Protect candidates, partner-removes-threat
credit, KO-capped expected damage, endgame logic (Protect only when something runs out), ally-only moves, Protect streak; their
Protect history and Tailwind/Trick Room turns left from the log; system prompt on how strong opponents play.
`scripts/live_scenarios.py` (exact live positions, no transcription): code 5/5, Opus 5/5.

### 5p. Oct 8 early: the simulator speaks the live dialect; measuring instead of guessing

- **Simulator parity (merged 04ffad4).** `sim/translate.py` now rebuilds every state from each seat's Showdown protocol
  log and requests, field for field like the live server (spread targets `[0]`, preview placeholders, `team` lists all
  six, opponent reveal rules, conditions keyed by the turn they started). `tests/test_sim_live_parity.py` (397 checks)
  pins it to the captures; nine deliberate drifts each fail it. Old agent code in the new simulator reproduces the
  live blindness (spread moves estimated 0/126, opponent sets blank 120/120); the fixed code: 123/123 and 0/120.
- **Head to head across commits** (`sim/subprocess_player.py`, `--p2 checkout:<path>`), code-only both sides:
  this branch beat 4f2d0cb (the version that lost live) **301-99 (75%)**. Then, each vs the previous commit over
  2,000 games: review fixes + switch-in preview 52%; joint attack scoring (one KO is one KO; focus fire knows the
  target can Protect) **56.4%**; damage valued in % of max HP, not current HP, **53.7%**.
- **Independent review** (second-opinion agent): Pixilate and Air Balloon gaps, Fake Out into Ghosts, Unburden and
  Prankster in the speed order, type labels for the model, switch-in judgement: all fixed. `scripts/live_scenarios.py`
  now 7 exact live positions: code 7/7, Opus 7/7 (3-13 s).
- **Were the four games winnable?** (`scripts/draft_eval.py`, `scripts/lineup_sweep.py`: the real drafted sixes in the
  simulator, our engine piloting both sides.) With the lineups actually played, our side wins 11% / 21% / 0% / 0.4%:
  our engine plays their teams better than the Oct 7 opponent did (M3: Fake Out on Great Tusk + Pixilate Hyper Voice turn 1,
  every game). Best of our 90 lineups vs their live four: M1 67%, M2 71%, M3 100%, M4 38% (best-of-90 is optimistic,
  except M3's 24/24); median lineup 4-17%. Verdict: every draft left us behind, Team Preview made three hopeless, and
  M4 was lost at the draft. Draft and Team Preview were the next work.
- **Draft v2 + Team Preview v2 + tier list (merged 57ced3f, from branch draft-v2).** `agent/pokemon/matchup.py`
  (1-on-1 races between complete sets, plain/Tailwind/Trick Room, on the real damage model) and a team model fitted on
  16,000 simulator games (`data/draft_model.json`); the drafter plays the rest of the snake out for every offered card
  against the opponent's best replies; Team Preview solves a 15x15 game of fours plus a lead sub-game that plays the
  leads' first two turns. `data/card_tiers.json` + `docs/card_tiers.md`: each live card's rating (its average worth on
  a team, in 1/100 log-odds), tier, why, and flags where our engine misplays a card (Trick Room setters, Prankster
  support, Incineroar's pivoting); the rating is the drafter's base value, so editing it moves the drafter by exactly
  that much. Written-down adjustments (Incineroar +25, Dragonite +17...) exist behind `draft_tier_adjust` (off).
  Branch results: v2 draft vs old 74.1% [71.3, 76.7] (1,000 games); v2 Team Preview vs old ~60% (800 games each).
  Draft pick median 28 ms, max 0.5 s (clock 15 s).
- **Measured after the merge** (live-format simulator, code-only both sides, `--p2 checkout:<worktree>`):

  | Us | Opponent | Result |
  |---|---|---|
  | this branch | 7be9183 (battle fixes, old draft + Team Preview) | 1,502 / 2,000 = **75.1%** |
  | this branch | 4f2d0cb (the version that lost 0-4 live) | 742 / 800 = **92.8%** |
  | this branch | search-bot-style counter-picker | 810 / 1,200 = **67.5%** |
  | 7be9183 | the same counter-picker | 418 / 1,200 = 34.8% |

  The four live pools re-drafted against the counter-picker: v2 88% vs old 84% (the simulated counter-picker drafts
  much worse than the Oct 7 opponent did, e.g. it never takes Gholdengo in the M1 pool, so this replay flatters both).
  Team Preview v2 on our live drafted sixes vs the Oct 7 opponent's actual lineups: 38% / 2% / 44% / 24% (we played 8 / 8 /
  0 / 4%; best of 90 was 67 / 71 / 100 / 38%). **Known weakness:** match 2: v2 still leads Iron Hands into Incineroar +
  Pixilate Sylveon (2%); its opening model rates that lead only mildly bad, and its prediction of their four leaves out
  Incineroar (rated low because our engine under-uses it). Two fixes tried and left off (predicting their four from the
  strong-player template; a robust lead rule). Opus, given the v2 candidates, picks v2's #1 in all four matches.
- A parallel session committed a README (9bd526d) and pushed `pokemon-agent` to origin at 00:15 Oct 8.

## 6. Milestones (Oct 6 → Oct 13)

| Day | Milestone | Done when |
|---|---|---|
| **Tue Oct 6** | **M0 Setup.** You: dashboard signup, Self Hosted, Official Agent Key into `.env`, Anthropic key + credits. Me: venv, `pip install -e ".[dev]" anthropic`, `pytest` (**done Oct 6: 498 passed**), `.env` scaffolded from the template (**done**), then `python -m agent --check-tournament` (all ✓) and a first test match with `examples.smoke_agent` once your keys are in. | Check shows ✓ on every line; one smoke match completes. |
| **Wed Oct 7** | **M1 Never-forfeit baseline.** `AnthropicProvider` + `FallbackProvider`, `MatchMemory`, `DecisionLog`, `PokemonAgent` that drafts with a trivial code rule, and uses the LLM for lineup and turns through the existing adapters. Raw-state capture to fixtures. **Code written Oct 6 (started early): 518 tests pass incl. 23 new. Remaining: first live match once keys are in.** | Plays a full match end to end with the model chain; log shows latency per call; zero `DecisionError`. |
| **Thu Oct 8** | **M2 Draft brain + data.** `scripts/build_dex.py` -> `data/*.json`; `TYPE_CHART`; `draft.py` scoring; memory stores all 12 cards. Unit tests. **Done Oct 6 (two days early): data vendored (1,481 species, 954 moves), type chart + level-50 stat math, opponent-aware draft scorer (offense/defense vs their actual sets, team coverage, shared weaknesses, support timing, denial). 547 tests pass.** | Draft picks in < 50 ms on fixtures; tests green; self-play draft looks sensible on review. |
| **Fri Oct 9** | **M3 Battle brain.** Speed order, damage estimates, threat flags, candidate ranking, the turn sheet in the prompt; `lineup.py` shortlist. **Code done Oct 6: `lineup.py` ranks all 15 lineups + leads; `battle.py` computes speed order (items, status, boosts, Trick Room), Gen 9 damage % incl. spread moves, opponent threats from known sets, and ranked candidate turns that double as the fallback. 556 tests. Remaining: validate against real server payloads in a live match.** | Turn sheet visible in logs; LLM agrees with code's #1 most turns; fallback path exercised by a forced provider error. |
| **Sat Oct 10** | **M4 Arena + tuning.** `arena.py`, `tally.py`, 10 self-play matches v-current vs v-M1 on Sonnet; prompt and heuristic fixes from the logs. Join open matches for real opponents. **Code done Oct 6: `agent/arena.py`, `scripts/tally.py`, `scripts/run_match.sh`. Remaining: the matches themselves.** | Tally shows current version ahead; p95 latency under 25 s on the judging model. |
| **Sun Oct 11** | **M5 Hardening.** Try/except wrapper around `choose_action` returning the smoke move; failure drills (§5.4); `run_tournament.sh` with `caffeinate -dims`; pull upstream if PR #5 merged and re-verify. **Done Oct 6: exception guard (tested), `scripts/run_tournament.sh` (pre-flight, caffeinate, auto-restart, tee'd log), upstream PR #5 merged in. Remaining: the live failure drills.** | All drills pass; restart mid-match resumes; no secrets in `git status`/`git log`. |
| **Mon Oct 12** | **M6 Validation.** 10–15 matches on the judging model (Opus 5.5) against the previous version and open opponents. Bug fixes only; no new features. | Win rate holds; zero bug-caused fallbacks. |
| **Tue Oct 13** | **Freeze.** Tag `v1.0`, push, make sure the commit is the latest on the repo's **default branch** (Official Rules §8), paste that full commit ID into the dashboard, `--check-tournament` ✓, PLAN.md updated with final numbers, redacted copy of `.env` saved next to `logs/`. | **Nothing is pushed after 11:59 p.m. PT** — not even a crash fix. The Rules allow a post-deadline edit only when Organizers authorize one in writing or a leaked secret must be removed. |
| Wed–Thu Oct 14–15 | Dry run from a cold laptop boot; register for the Pokémon tournament as soon as it appears on the dashboard. | Registered; dry run clean. |

---

## 7. Tournament-day checklist (Fri Oct 16)

**Night before**
- [ ] Laptop updates done, auto-update off, sleep off, screensaver off, charger packed.
- [ ] `python -m agent --check-tournament` shows ✓ on every line.
- [ ] Anthropic console: credits ≥ $50 (a match costs ~$1.50 on Opus; 11 games ≈ $20), no rate-limit warnings. Note status page URL.
- [ ] `.env` has the Official Agent Key, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` and `AGENT_MODEL=claude-opus-5-5` (`grep ^AGENT_MODEL .env`; saves from an open TextEdit window reverted this line twice on Oct 7); `git status` clean; `.env` untracked.
- [ ] Phone hotspot tested as backup network.

**09:00–09:45**
- [ ] Plug in. Open two terminals in the repository checkout, venv active in both.
- [ ] Terminal 1: `./scripts/run_tournament.sh` (verified Oct 7: pre-flight, `caffeinate -dims python -m agent --tournament`, auto-restart after a kill; it also captures raw states to `tests/fixtures/live/`). Wait for `Connected with your Official Agent Key.` then `Waiting for your next game...`. Do NOT add `--match` on tournament day and run only one copy.
- [ ] Terminal 2: `tail -f logs/*.jsonl` for the live decision log.
- [ ] Dashboard open: confirm registration shows the agent as Self Hosted and ready. Then **close any spectator/replay view**: Rules §6 says the person who can restart the agent must not watch its match through a spectator view.

**10:00–12:00**
- [ ] Do not edit code. Do not restart unless the process is dead.
- [ ] If it dies: `run_tournament.sh` restarts it by itself (preconfigured fault recovery, allowed by Rules §4). If you ever restart it by hand, write down the time and reason: that is a manual intervention and belongs in the records (Rules §9).
- [ ] If Opus errors spike (watch the `model` and `provider_errors` fields in the log): nothing to do, the chain falls to Sonnet, then code.
- [ ] If Wi-Fi drops: switch to hotspot; restart the runtime if it did not recover within 60 s. Remember the 4-minute connect window per game.

**After**
- [ ] Copy `logs/` somewhere safe and keep it, with the submitted commit ID and a redacted `.env`, for at least 30 days after results (Rules §8–9). Rotate the Official Agent Key in the dashboard if anyone saw your screen.

---

## 8. Steps only you can do

1. **Dashboard signup** (link posted on the tournament Discord, opens today Oct 6). Choose **Self Hosted**.
2. **Complete event registration** on the dashboard (questionnaire, Rules, and a repository field; use `https://github.com/holmjames/ace-trainer-claude` — renamed Oct 7, the old URL redirects). The "Register my agent" button stays disabled until this is complete.
3. **Generate the Official Agent Key** on Agent Configuration. It is shown **once**. Paste it into the checkout's `.env` as `ALTRUAGENT_OFFICIAL_AGENT_KEY=...`. Never paste it into chat with me, a commit, or a screenshot.
4. **Anthropic API key**: ~~create one~~ **done Oct 6.** Key plus `ANTHROPIC_WORKSPACE_ID` are in `.env` and a live Fable 5.1 call succeeded (about 4.6 s per small call). Still to confirm in the Console: prepaid credits loaded, standard data retention. Lesson learned: close TextEdit after editing `.env`, a later save from an open window silently reverted lines added by scripts.
5. **Register my agent** for the Pokémon tournament on the Tournaments page when it appears (likely Oct 14–15). Not Werewolf, not Red Alert.
6. **Create test matches** on the Testing page when we test (I can't click the dashboard; you create, I run).
7. **Be at the laptop** Oct 16 from 09:00 with power and a hotspot.

---

## 9. Open questions (answered Oct 6)

1. **Budget**: **$250 approved.** Testing can run on Fable 5.1 as well as Sonnet 5.5; we still log cost per match and switch to Sonnet for bulk runs if the total climbs past ~$150 before validation week.
2. **Fork visibility**: **public fork is fine.** Secrets check before every push stays mandatory.
3. **Showdown data**: **approved.** `scripts/build_dex.py` vendors species and move tables from Pokémon Showdown's open-source data into `data/`.
4. **Python**: resolved, 3.14 works (see §2).
5. **Discord**: James is locating the tournament Discord (Oct 6). Once found: watch for the signup link, format announcements, and the upstream PR #5 merge.

---

## 10. Verification (how we know it works, end to end)

- `pytest` green at every milestone, including `tests/test_live_fixtures.py` and `tests/test_live_semantics.py`, which replay every real captured state (draft, lineup, turn) through the code-only agent and check that each gets a computed, well-formed decision without the exception path, and what the agent knew at each decision.
- `python -m agent --check-tournament` ✓ before M1, after M5, on Oct 13, and on the morning of Oct 16.
- Every test match leaves a `logs/<session_id>.jsonl`; `scripts/tally.py` summarizes wins, latency p95, provider mix and fallback reasons. Promotion of a version requires the numbers in §5.3.
- Failure drills in §5.4 pass on Oct 11 and again on the dry run Oct 14–15.
- Secrets check before every push: `git status` shows `.env` untracked and `grep -r "eak_live_\|sk-ant" --exclude-dir=.git .` returns nothing.
