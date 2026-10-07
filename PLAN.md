# PLAN.md — UCLA AI Agent Gaming Tournament: Pokémon VGC Doubles Draft agent

Approved by James on Oct 6, 2026. No agent code has been written yet; work starts at Milestone 0.

Dates: today is **Mon Oct 6, 2026**. Code freeze **Mon Oct 13**. Tournament **Thu Oct 16, 10:00–12:00 PDT**.

---

## 0. Context (why this plan looks the way it does)

- You chose **Pokémon Showdown only**. Werewolf is dropped entirely.
- The tournament's Pokémon game is `pokemon_vgc_doubles_draft`: two agents snake-draft 6 Pokémon
  each from a shared 18-card pool, pick 4 of 6 at Team Preview, then play a 4v4 doubles battle.
- Decisions by **Claude Fable 5.1** (`claude-fable-5-1`) through the official `anthropic` Python
  SDK, key in `.env` as `ANTHROPIC_API_KEY`. Fallback chain: Sonnet 5.5, then a code-only move.
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
  you after 300 s). Our design makes this unreachable: every return value passes the validators in
  `examples/llm/pokemon.py` before it leaves our code.

---

## 2. What the docs say that shapes the plan (verified Oct 6)

| Topic | Fact | Consequence |
|---|---|---|
| Draft clock | **15 s per pick**, measured from when it becomes your turn and *including* any model call. Timeout = server picks a random card; late picks are rejected `STALE_STATE`. | **Draft is pure code.** No LLM call in the draft, ever. Fable 5.1 always thinks and can take 5–20 s. |
| Battle clock | **300 s** per Team Preview or turn decision; timeout = random legal move. | LLM is fine here. Hard client timeout 40 s on Fable, 12 s on Sonnet, then code. |
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
| Fable 5.1 API | Thinking is always on (omit the `thinking` param); depth via `output_config.effort`; structured JSON via `output_config.format` (`json_schema`); can return `stop_reason: "refusal"`; not available to zero-data-retention orgs. | Provider sets `effort: "low"` for speed, checks `stop_reason`, uses a 40 s timeout and `max_retries=0` so we own retries and fallback. Default Anthropic orgs are fine; confirm yours is not ZDR. |

---

## 3. Pokémon vs Werewolf (closed: you chose Pokémon)

Short record of why that is also the right call:
- Pokémon's draft and battle have **perfect information about builds** and well-defined math, so code can carry most of the weight and the LLM adds judgment on top. Werewolf's edge was almost entirely prompting, under a 120 s shared chat window with 7 agents.
- Pokémon's 300 s turn clock is forgiving of a slow, careful model. Only the 15 s draft is tight, and code handles it.
- Pokémon has the bigger, deeper bracket (6 Swiss rounds + top 32 best-of-3), so a consistent agent gets rewarded.

---

## 4. Architecture: hybrid, code first, LLM on top

```
altruagent-starter/
  agent/
    agent.py              # create_agent() -> PokemonAgent (thin router by state.phase)
    llm/
      anthropic_provider.py   # AnthropicProvider (official SDK) + FallbackProvider chain
    pokemon/
      memory.py           # MatchMemory: both rosters' full card sets, turn history, revealed info
      log.py              # DecisionLog: one JSON line per decision -> logs/<session_id>.jsonl
      data.py             # TYPE_CHART (18x18), species/move lookups from data/*.json
      draft.py            # PURE CODE: score_pool(...) -> best card_id in < 50 ms
      lineup.py           # code shortlist of lineups + one LLM pick (300 s clock)
      battle.py           # code: damage/speed/threat table + candidate ranking; LLM picks; validated
      prompts.py          # system prompt + the JSON schemas the model must answer in
    arena.py              # self-play helper: picks agent version by seat (see §6)
  data/
    pokedex.json, moves.json   # vendored from Pokémon Showdown data (MIT), built by scripts/build_dex.py
  logs/                   # gitignored; decision logs from every test match
  scripts/
    build_dex.py          # regenerates data/*.json
    tally.py              # win rate, latency p95, fallback rate per version from logs/
    run_tournament.sh     # caffeinate + python -m agent --tournament, restart on exit
  tests/
    test_draft.py, test_battle_math.py, test_provider.py, test_payloads.py
```

**Reuse, don't rewrite** (all already in the repo):
- `examples/llm/pokemon.py` — `lineup_choice` and `doubles_choice` already turn the server's template into a bounded choice and **validate** the model's answer (`build()` rejects wrong species, bad targets, double-switch into one Pokémon, double pass). We import these and route every lineup/turn answer through their `build()`.
- `examples/llm/base.py` — `Choice`, `InvalidChoice`, `object_schema` (strict JSON schema builder).
- `examples/smoke_agent.choose_action` — a deterministic, always-legal move for any Pokémon phase. This is the **last-resort fallback** everywhere.
- `examples/llm_agent.py` — the `_ask` retry-once-with-the-error pattern and the `LLMProvider` protocol (`complete_structured(messages, schema_name, schema) -> dict`). We implement that protocol, so the rest of the example machinery works unchanged.
- `tests/test_llm_agent.py` — `FakeProvider` and the Pokémon state fixtures; copy their shape for our tests.

### 4.1 The decision flow per phase

**Draft (code only, must return in well under 1 s)**
Score every offered card with `score_pool(card, my_roster, opp_roster, picks_left)`:
- base stat total and speed tier (from `data/pokedex.json`, falling back to the card if present);
- **type coverage**: reward offensive types we lack and defensive resistances to what the opponent has already drafted; penalize stacking a shared weakness;
- role balance: physical vs special attackers, at least one speed-control/support set (Fake Out, Tailwind, Trick Room, Icy Wind, redirection) by pick 4–5;
- **denial**: small bonus for taking a card that best completes the opponent's roster when our own top two scores are close;
- a deterministic tiebreak so the same pool always drafts the same way (reproducible tests).
Store the **full card** of every pick, ours and theirs, in `MatchMemory`.

**Team Preview (code shortlist, LLM chooses, 300 s)**
Code enumerates the 15 four-Pokémon subsets, scores each against the opponent's 6 (coverage, speed, known threats from their drafted moves), proposes leads (fastest/Fake Out/weather setter pairs), and keeps the top 3. The LLM sees the 3 options with the computed numbers and the opponent's exact sets, and picks one. Answer goes through `lineup_choice.build()`. Timeout or error -> code's #1.

**Battle turn (code computes, LLM judges, 300 s)**
Code builds a compact **turn sheet** every turn:
- speed order of the 4 active Pokémon (base stats + known EVs/nature, boosts, Tailwind/Trick Room/paralysis);
- for each of our legal moves vs each target: type multiplier, STAB, rough damage as a % of the target's current HP (Gen 9 formula, level 50, using the opponent's known EVs/nature/item from the draft);
- the opponent's **known** moves per active Pokémon (from their drafted cards), with their rough damage to us, and flags like "can OHKO slot 0", "has Protect", "has Fake Out", "priority move";
- a ranked list of 3–5 candidate full turns (both slots), e.g. "double into opp A", "Protect slot 0 + attack", "switch slot 1 to resist".
The LLM sees the sheet plus the raw per-slot options from the server and returns `{slot_0, slot_1, reasoning_summary}` in the exact schema `doubles_choice` expects. `doubles_choice.build()` validates it. Any failure -> retry once with the error -> candidate #1 -> `smoke_agent` move.

### 4.2 Per-match memory and decision log
- `MatchMemory` (one per process, created in `create_agent()`): `my_cards`, `opp_cards` (full sets by species), `first_drafter`, `my_lineup`, `opp_lineup_seen`, per-turn `turns[]` (what we played, what the opponent did, HP after), `fallbacks_used`, latencies.
- `DecisionLog` appends one JSON line per decision to `logs/<session_id>.jsonl`: phase, turn, compact state summary, the turn sheet, candidates, the model's answer, which provider answered (fable / sonnet / code), latency ms, validation errors, and the final payload. **Never** any key, token, or env value. This is what we read after every test match to answer "why did it do that".

### 4.3 The Anthropic provider and fallback chain
`AnthropicProvider.complete_structured(messages, schema_name, schema)`:
- `anthropic.Anthropic(timeout=<per-provider>, max_retries=0)`; the SDK reads `ANTHROPIC_API_KEY` from the environment after `load_dotenv()`. The key is never stored on our object, never in `repr`, never logged. James's key is a personal key spanning several workspaces, so every request also carries the `anthropic-workspace-id` header, read from `ANTHROPIC_WORKSPACE_ID` in `.env` (verified live Oct 6).
- Request: `model`, `max_tokens≈2000`, `system` (stable, marked `cache_control` ephemeral so repeated turns get the cache discount), `messages=[{"role":"user", ...}]`, `output_config={"format": {"type": "json_schema", "schema": schema}, "effort": "low"}`. No `thinking` parameter (Fable 5.1 thinks by default; sending `disabled` or a budget is a 400).
- Check `stop_reason`: `"refusal"` or `"max_tokens"` -> `ProviderError`. Parse the first text block as JSON; non-object -> `ProviderError`.
- Errors are mapped most-specific-first (`RateLimitError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError`) to `ProviderError` with the class name only, never the body.
- Server-side refusal fallback (`fallbacks: "default"` with the beta header) is available but **not** used: our own chain already covers refusals and keeps one code path. Can be added later if logs show refusals.

`FallbackProvider([Fable 5.1 @ 40 s, Sonnet 5.5 @ 12 s])` tries each in order and raises only if all fail; the caller then uses code. Model ids come from `.env` (`AGENT_MODEL=claude-fable-5-1`, `AGENT_FALLBACK_MODEL=claude-sonnet-5-5`) so we can run cheap test matches on Sonnet without code changes.

### 4.4 Rough cost per match
Prices: Fable 5.1 $10 in / $50 out per million tokens; Sonnet 5.5 $2 / $10. A battle decision sends roughly 4–6K input tokens (system prompt cached after the first turn) and produces roughly 1–2K output tokens including low-effort thinking.

| Item | Calls per match | Fable 5.1 | Sonnet 5.5 |
|---|---|---|---|
| Draft | 0 (code) | $0 | $0 |
| Team Preview | 1 | ~$0.12 | ~$0.03 |
| Battle turns | ~12–20 | ~$0.10–0.15 each | ~$0.02–0.03 each |
| **Per match** | | **~$2–3** | **~$0.40–0.60** |
| Tournament day (6–11 games) | | **~$15–35** | |
| Testing (40–60 matches) | | ~$100–150 on Fable, ~$25 on Sonnet | |

Budget approved: **$250**. Default to Fable 5.1 for testing so what we tune is what we play; drop to Sonnet 5.5 for bulk self-play if spend passes ~$150 before validation week.

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
- **Effort sweep (Oct 6, three parallel runs of the 9 scenarios):** low 9/9, median ~3 s, max 16 s; medium 9/9, max 22 s; high 9/9, max 33 s (one call fell through to Sonnet 5.5, which also answered correctly). Decision: **stay on `low`**. It is the fastest and cheapest and lost nothing; the 300 s clock leaves huge margin either way. Revisit only if live matches show judgment errors the sheet can't fix.
- Still to do offline: harder scenarios as live matches reveal mistakes (the current nine no longer discriminate between effort levels).

## 6. Milestones (Oct 6 → Oct 13)

| Day | Milestone | Done when |
|---|---|---|
| **Mon Oct 6** (today) | **M0 Setup.** You: dashboard signup, Self Hosted, Official Agent Key into `.env`, Anthropic key + credits. Me: venv, `pip install -e ".[dev]" anthropic`, `pytest` (**done Oct 6: 498 passed**), `.env` scaffolded from the template (**done**), then `python -m agent --check-tournament` (all ✓) and a first test match with `examples.smoke_agent` once your keys are in. | Check shows ✓ on every line; one smoke match completes. |
| **Tue Oct 7** | **M1 Never-forfeit baseline.** `AnthropicProvider` + `FallbackProvider`, `MatchMemory`, `DecisionLog`, `PokemonAgent` that drafts with a trivial code rule, and uses the LLM for lineup and turns through the existing adapters. Raw-state capture to fixtures. **Code written Oct 6 (started early): 518 tests pass incl. 23 new. Remaining: first live match once keys are in.** | Plays a full match end to end with Fable; log shows latency per call; zero `DecisionError`. |
| **Wed Oct 8** | **M2 Draft brain + data.** `scripts/build_dex.py` -> `data/*.json`; `TYPE_CHART`; `draft.py` scoring; memory stores all 12 cards. Unit tests. **Done Oct 6 (two days early): data vendored (1,481 species, 954 moves), type chart + level-50 stat math, opponent-aware draft scorer (offense/defense vs their actual sets, team coverage, shared weaknesses, support timing, denial). 547 tests pass.** | Draft picks in < 50 ms on fixtures; tests green; self-play draft looks sensible on review. |
| **Thu Oct 9** | **M3 Battle brain.** Speed order, damage estimates, threat flags, candidate ranking, the turn sheet in the prompt; `lineup.py` shortlist. **Code done Oct 6: `lineup.py` ranks all 15 lineups + leads; `battle.py` computes speed order (items, status, boosts, Trick Room), Gen 9 damage % incl. spread moves, opponent threats from known sets, and ranked candidate turns that double as the fallback. 556 tests. Remaining: validate against real server payloads in a live match.** | Turn sheet visible in logs; LLM agrees with code's #1 most turns; fallback path exercised by a forced provider error. |
| **Fri Oct 10** | **M4 Arena + tuning.** `arena.py`, `tally.py`, 10 self-play matches v-current vs v-M1 on Sonnet; prompt and heuristic fixes from the logs. Join open matches for real opponents. **Code done Oct 6: `agent/arena.py`, `scripts/tally.py`, `scripts/run_match.sh`. Remaining: the matches themselves.** | Tally shows current version ahead; p95 latency under 25 s on Fable. |
| **Sat Oct 11** | **M5 Hardening.** Try/except wrapper around `choose_action` returning the smoke move; failure drills (§5.4); `run_tournament.sh` with `caffeinate -dims`; pull upstream if PR #5 merged and re-verify. **Done Oct 6: exception guard (tested), `scripts/run_tournament.sh` (pre-flight, caffeinate, auto-restart, tee'd log), upstream PR #5 merged in. Remaining: the live failure drills.** | All drills pass; restart mid-match resumes; no secrets in `git status`/`git log`. |
| **Sun Oct 12** | **M6 Validation.** 10–15 matches on Fable 5.1 against the previous version and open opponents. Bug fixes only; no new features. | Win rate holds; zero bug-caused fallbacks. |
| **Mon Oct 13** | **Freeze.** Tag `v1.0`, push to your fork, `--check-tournament` ✓, PLAN.md updated with final numbers. | Nothing changes after today except a confirmed crash fix. |
| Tue–Wed Oct 14–15 | Dry run from a cold laptop boot; register for the Pokémon tournament as soon as it appears on the dashboard. | Registered; dry run clean. |

---

## 7. Tournament-day checklist (Thu Oct 16)

**Night before**
- [ ] Laptop updates done, auto-update off, sleep off, screensaver off, charger packed.
- [ ] `python -m agent --check-tournament` shows ✓ on every line.
- [ ] Anthropic console: credits ≥ $50, no rate-limit warnings. Note status page URL.
- [ ] `.env` has the Official Agent Key and `ANTHROPIC_API_KEY`; `git status` clean; `.env` untracked.
- [ ] Phone hotspot tested as backup network.

**09:00–09:45**
- [ ] Plug in. Open two terminals in `altruagent-starter`, venv active in both.
- [ ] Terminal 1: `./scripts/run_tournament.sh` (runs `caffeinate -dims python -m agent --tournament`, restarts if it exits). Wait for `Connected as official tournament agent.`
- [ ] Terminal 2: `tail -f logs/*.jsonl` for the live decision log.
- [ ] Dashboard open: confirm registration shows the agent as Self Hosted and ready.

**10:00–12:00**
- [ ] Do not edit code. Do not restart unless the process is dead.
- [ ] If it dies: re-run the same command; the runtime re-authenticates and resumes the active game.
- [ ] If Fable errors spike (watch `provider` field in the log): nothing to do, the chain falls to Sonnet, then code.
- [ ] If Wi-Fi drops: switch to hotspot; restart the runtime if it did not recover within 60 s. Remember the 4-minute connect window per game.

**After**
- [ ] Copy `logs/` somewhere safe. Rotate the Official Agent Key in the dashboard if anyone saw your screen.

---

## 8. Steps only you can do

1. **Dashboard signup** (link posted on the tournament Discord, opens today Oct 6). Choose **Self Hosted**.
2. **Complete event registration** on the dashboard (questionnaire, Rules, and a repository field; use `https://github.com/holmjames/altruagent-starter`). The "Register my agent" button stays disabled until this is complete.
3. **Generate the Official Agent Key** on Agent Configuration. It is shown **once**. Paste it into `altruagent-starter/.env` as `ALTRUAGENT_OFFICIAL_AGENT_KEY=...`. Never paste it into chat with me, a commit, or a screenshot.
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

- `pytest` green at every milestone, including `test_payloads.py`, which runs every code path (draft, lineup, turn, each fallback) against real captured fixtures and asserts the output passes `examples/llm/pokemon.py`'s validators.
- `python -m agent --check-tournament` ✓ before M1, after M5, on Oct 13, and on the morning of Oct 16.
- Every test match leaves a `logs/<session_id>.jsonl`; `scripts/tally.py` summarizes wins, latency p95, provider mix and fallback reasons. Promotion of a version requires the numbers in §5.3.
- Failure drills in §5.4 pass on Oct 11 and again on the dry run Oct 14–15.
- Secrets check before every push: `git status` shows `.env` untracked and `grep -r "eak_live_\|sk-ant" --exclude-dir=.git .` returns nothing.
