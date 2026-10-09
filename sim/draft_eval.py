"""Draft and Team Preview experiments on the local engine: code-only on both sides, $0, minimal logs.

    python sim/draft_eval.py match v2 v1 --pairs 500 --seed 1          # new drafter vs old drafter
    python sim/draft_eval.py match v2 counter --pairs 500 --seed 2     # new drafter vs a search-bot-style counter-picker
    python sim/draft_eval.py collect random random --games 10000 --out /tmp/x.jsonl   # training data for the draft model
    python sim/draft_eval.py live v2 counter --games 200       # the four Oct 7 pools, our drafter in our live seat
    python sim/draft_eval.py preview --games 200                # Team Preview choices vs their live lineups (M2, M3)
    python sim/draft_eval.py match v2l checkout:<path>:checkout_shim:<python> --pairs 150   # paired vs another checkout
    python sim/draft_eval.py match v2l v1 --pairs 200 --seed 7 --pool-has Archaludon   # only pools holding that card
    python sim/draft_eval.py match 'v2l{"draft_tiebreak_exposure": 1}' v1 --pairs 200 --seed 11 --pin-tiebreak   # arms share our tie-break draws
    python sim/draft_eval.py match v2l v1 --pairs 200 --seed 9 --pool-has Archaludon --first-only b   # only the game B drafts first

Players (``PLAYERS`` below): ``v1`` is the old greedy scorer with the old Team Preview, ``v2`` the counter-aware
drafter (old Team Preview), ``v2l`` the counter-aware drafter with the new Team Preview, ``v1l`` the old drafter
with the new Team Preview, ``counter`` the counter-picker (agent/versions/counter_picker.py, old Team Preview),
``random`` a seeded random drafter. ``name{"key": value}`` adds tuning overrides, e.g. ``v2{"draft_lookahead": 0}``.
For those the battle itself is always the code brain (agent/pokemon/battle.py), so only drafting and Team Preview
differ. A sim/harness.py spec — ``checkout:<path>[:<module>[:<python>]]`` (an agent from another checkout, e.g. the
Oct 7 opponent's through sim/shims/checkout_shim.py, in a child process) or a ``module[:factory]`` path — plays with
ITS own battle brain; ``fable`` is agent/agent.py with the model chain from this checkout's .env (the orchestrator's
Opus-piloted games; ``--log-dir`` keeps their decision logs). Every ``--out`` line carries the game's wall-clock
``seconds``, each seat's deciding time per decision, and the clock events when ``--clock`` is on.

``match`` plays games in PAIRS: the same 18-card pool and the same battle seed twice, with the two players swapping
seats, so each drafts first exactly once per pool. That cancels most of the pool and first-pick luck. Output: win
rate of the first player with a 95% Wilson interval, overall and by who drafted first. ``--pool-has <Species>`` plays
only the pools holding that card: seeds without it are skipped, deterministically, so every arm run with the same
``--seed`` plays the same pools in the same seats (``--pairs`` then counts the pools played). ``--first-only a`` (or
``b``) plays only the game of each pool in which player A (or B) drafts first; ``--pairs`` still counts pools.

``--pin-tiebreak`` pins the secret tie-break seed (agent/pokemon/draft.py ``tiebreak_seed``) of the in-process players
game by game: ``draft._SEEDS`` is cleared and ``draft.secrets.randbits`` returns bits of sha256(f"{battle seed}:{swap}")
(swap False / True, the seat order) for that game only, as scripts/preview_replay.py does for one session. Two arms run
on the same ``--seed`` then make the same draft and Team Preview draws in every game and differ only where their
settings act. A ``checkout:`` player runs in its own process and keeps its own randomness.

Decision logs are switched off (the agent's DecisionLog is pointed at /dev/null) and each game appends one short
JSON line to ``--out`` (default: none), so thousands of games cost a few megabytes at most. One engine process; do
not run two of these at once (two bridges on one machine have deadlocked before).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon import data, tuning  # noqa: E402
from sim import harness  # noqa: E402
from sim.clock import Clock  # noqa: E402

NO_LOG = "/dev/null"  # DecisionLog cannot create a directory there, so it stays silent (by design it never raises)
LOG_DIR = NO_LOG  # where in-process agents write decision logs (``--log-dir``; off by default)

# name -> tuning overrides on top of the defaults (and, for some, a different agent class)
PLAYERS: dict[str, dict] = {
    "v1": {"draft_v2": 0, "lineup_v2": 0},
    "v2": {"draft_v2": 1, "lineup_v2": 0},
    "v2l": {"draft_v2": 1, "lineup_v2": 1},
    "v1l": {"draft_v2": 0, "lineup_v2": 1},
    "counter": {"draft_v2": 0, "lineup_v2": 0},
    "counterl": {"draft_v2": 0, "lineup_v2": 1},
    "random": {"draft_v2": 0, "lineup_v2": 0},
}


def external(spec: str) -> bool:
    """A sim/harness.py player rather than one of ``PLAYERS``: ``checkout:<path>[:<module>[:<python>]]``, a module path, or
    ``fable`` (agent/agent.py with the model chain from the .env of the checkout this script runs from)."""
    return spec.startswith("checkout:") or "." in spec.partition("{")[0] or spec in ("fable", "code")


def parse_spec(spec: str) -> tuple[str, dict | None]:
    """'v2{"draft_lookahead": 0}' -> ('v2', {...}); a sim/harness.py spec -> (spec, None). Unknown names are an error."""
    name, brace, rest = spec.partition("{")
    if name in PLAYERS:
        extra = json.loads(brace + rest) if brace else {}
        return name, {**PLAYERS[name], **extra}
    if external(spec):
        return spec, None
    raise SystemExit(f"unknown player {name!r}; known: {sorted(PLAYERS)}, checkout:<path>[:<module>[:<python>]] or a module path")


class _SeededRandomDraft(PokemonAgent):
    """Random draft picks from a seeded generator (reproducible training data); everything else is the code brain."""

    def __init__(self, seed: int, params: dict) -> None:
        super().__init__(None, version="random-draft", params=params, log=lambda _line: None)
        self._rng = random.Random(seed)

    def _draft(self, state, obs, log):
        pick = self._rng.choice(state.legal_actions)
        from agent.pokemon import draft as draft_rules

        self.memory.record_pick(draft_rules.card_id_of(pick))
        return pick, "Random draft pick."


def make_player(spec: str, seed: int):
    name, overrides = parse_spec(spec)
    if overrides is None:  # sim/harness.py's spelling: another checkout's agent in a child, a module path or fable in-process
        os.environ["AGENT_LOG_DIR"] = LOG_DIR  # a child of our own code reads it; set before the child is spawned
        agent, _ = harness.make_player(spec, random.Random(seed))
        if hasattr(agent, "_log_dir"):
            agent._log_dir = LOG_DIR
        return agent
    params = tuning.merged(overrides)
    if name == "random":
        agent = _SeededRandomDraft(seed, params)
    elif name.startswith("counter"):
        from agent.versions.counter_picker import CounterPickerAgent

        agent = CounterPickerAgent(params=params)
        agent._print = lambda _line: None
    else:
        agent = PokemonAgent(None, version=spec, params=params, log=lambda _line: None)
    agent._log_dir = LOG_DIR
    return agent


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Win rate and its 95% Wilson interval (better than the normal approximation near 0 or 1)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


class _Timed:
    """Counts the wall-clock seconds a seat spends deciding (its agent's ``choose_action`` calls), per decision too."""

    def __init__(self, inner, spec: str) -> None:
        self.inner, self.spec, self.seconds = inner, spec, 0.0
        self.decisions: list[float] = []

    @property
    def _version(self):
        return getattr(self.inner, "_version", getattr(self.inner, "name", self.spec))

    def choose_action(self, state, context):
        t0 = time.monotonic()
        try:
            return self.inner.choose_action(state, context)
        finally:
            elapsed = time.monotonic() - t0
            self.seconds += elapsed
            self.decisions.append(round(elapsed, 3))


def play(bridge, game_id: str, seed: int, specs: dict[str, str], names: dict[str, str], *, clock: bool = False) -> dict:
    rng = random.Random(seed)
    players = {side: _Timed(make_player(spec, seed * 2 + (side == "p2")), spec) for side, spec in specs.items()}
    t0 = time.monotonic()
    result = harness.play_game(bridge, game_id, players, names, rng, verbose=False, clock=Clock() if clock else None)
    result["seconds"] = round(time.monotonic() - t0, 3)
    result["seat_seconds"] = {side: round(p.seconds, 3) for side, p in players.items()}
    result["decision_seconds"] = {side: list(p.decisions) for side, p in players.items()}
    return result


def record(out, result: dict, seed: int, specs: dict[str, str]) -> None:
    if out is None:
        return
    out.write(json.dumps({"seed": seed, "specs": specs, "first": result["first_drafter"], "winner": result["winner_side"],
                          "turns": result["turns"], "seconds": result.get("seconds"), "seat_seconds": result.get("seat_seconds"),
                          "decision_seconds": result.get("decision_seconds"), "forfeit": result.get("forfeit"), "clock": result.get("clock"),
                          "rosters": result["rosters"], "teams": result["teams"], "picks": result.get("picks")}) + "\n")
    out.flush()


def rate(played: int, started: float) -> str:
    elapsed = time.monotonic() - started
    return f"{60 * played / elapsed:.1f} games/min" if elapsed > 0 and played else "no games"


def pool_of(seed: int) -> tuple[list[dict], str]:
    """The 18-card pool and the first drafter ("p1" / "p2") that ``match`` plays for battle seed ``seed``:
    harness.play_game's first two draws from ``random.Random(seed)``, made without playing."""
    rng = random.Random(seed)
    pool = rng.sample(harness.CARDS, 18)
    return pool, rng.choice(["p1", "p2"])


def match_seeds(seed: int, pairs: int, pool_has: str | None = None) -> list[int]:
    """The battle seeds ``match`` plays, in order: ``seed * 1_000_003 + k`` for k = 0, 1, 2, ... With ``pool_has`` (a
    species) every seed whose pool lacks it is skipped until ``pairs`` pools hold it: the same pools for every arm."""
    if not pool_has:
        return [seed * 1_000_003 + k for k in range(pairs)]
    want = data.to_id(pool_has)
    if not any(data.to_id(c["species"]) == want for c in harness.CARDS):
        raise SystemExit(f"--pool-has {pool_has!r}: no such species in sim/cards.json")
    seeds: list[int] = []
    k = 0
    while len(seeds) < pairs:
        if any(data.to_id(c["species"]) == want for c in pool_of(seed * 1_000_003 + k)[0]):
            seeds.append(seed * 1_000_003 + k)
        k += 1
    return seeds


def pinned_seed(seed: int, swap: bool) -> int:
    """``--pin-tiebreak``: the 64-bit tie-break seed of the game on battle seed ``seed``, seats swapped or not."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{swap}".encode()).digest()[:8], "big")


@contextlib.contextmanager
def pin_tiebreak(seed: int, swap: bool):
    """For one game: an empty seed registry and ``draft.secrets.randbits`` answering from ``pinned_seed(seed, swap)``,
    so every in-process player's match seed is that value; the real ``randbits`` is back afterwards, even on an error."""
    from agent.pokemon import draft as draft_rules

    value = pinned_seed(seed, swap)
    original = draft_rules.secrets.randbits
    draft_rules._SEEDS.clear()  # keyed by id(my_cards), which a later game's dict can reuse
    draft_rules.secrets.randbits = lambda bits: value & ((1 << bits) - 1)
    try:
        yield value
    finally:
        draft_rules.secrets.randbits = original
        draft_rules._SEEDS.clear()


def cmd_match(args) -> int:
    global LOG_DIR
    LOG_DIR = getattr(args, "log_dir", None) or NO_LOG
    a, b = args.a, args.b
    pool_has = getattr(args, "pool_has", None)
    pin = bool(getattr(args, "pin_tiebreak", False))
    first_only = getattr(args, "first_only", None)
    seeds = match_seeds(args.seed, args.pairs, pool_has)
    if pool_has and seeds:
        print(f"--pool-has {pool_has}: {len(seeds)} pools from seeds {args.seed * 1_000_003}..{seeds[-1]} "
              f"({seeds[-1] - args.seed * 1_000_003 + 1 - len(seeds)} without it skipped)", flush=True)
    if pin:
        print("--pin-tiebreak: in-process tie-break seeds pinned per game (sha256 of battle seed and seat swap)", flush=True)
    if first_only:
        print(f"--first-only {first_only}: only the game of each pool in which {a if first_only == 'a' else b} drafts first", flush=True)
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    bridge = harness.Bridge()
    wins = games = played = 0
    by_first = {"a_first": [0, 0], "b_first": [0, 0]}
    errors = timeouts = forfeits = 0
    started = time.monotonic()
    try:
        for k, seed in enumerate(seeds):
            first = pool_of(seed)[1] if first_only else None
            for swap in (False, True):
                if first_only and (first == ("p2" if swap else "p1")) != (first_only == "a"):
                    continue  # --first-only: the other player drafts first in this seat order
                specs = {"p1": b, "p2": a} if swap else {"p1": a, "p2": b}
                names = {"p1": "B" if swap else "A", "p2": "A" if swap else "B"}
                try:
                    with pin_tiebreak(seed, swap) if pin else contextlib.nullcontext():
                        result = play(bridge, f"eval-{seed}-{int(swap)}", seed, specs, names, clock=bool(getattr(args, "clock", False)))
                except Exception as exc:  # noqa: BLE001 — one broken game must not end a long run
                    errors += 1
                    print(f"   pair {k} swap {swap}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                played += 1
                if result.get("clock"):
                    timeouts += sum(c["timeouts"] for c in result["clock"].values())
                    forfeits += bool(result.get("forfeit"))
                record(out, result, seed, specs)
                if result["winner"] is None:
                    continue  # unfinished (turn cap): counts for neither
                games += 1
                a_won = result["winner"] == "A"
                wins += a_won
                a_side = "p2" if swap else "p1"
                bucket = by_first["a_first" if result["first_drafter"] == a_side else "b_first"]
                bucket[0] += a_won
                bucket[1] += 1
            if args.progress and (k + 1) % args.progress == 0:
                p, lo, hi = wilson(wins, games)
                print(f"  {k + 1} pairs: {a} {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] over {games} games "
                      f"({int(time.monotonic() - started)} s)", flush=True)
    finally:
        bridge.close()
        if out:
            out.close()
    p, lo, hi = wilson(wins, games)
    print(f"{a} vs {b}: {wins}/{games} = {100 * p:.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]  "
          f"({errors} errors, {int(time.monotonic() - started)} s, {rate(played, started)})"
          + (f"; clock: {timeouts} timeouts, {forfeits} forfeits" if getattr(args, "clock", False) else ""))
    for key, (w, n) in by_first.items():
        q, l, h = wilson(w, n)
        print(f"   {key:8s} {w}/{n} = {100 * q:.1f}% [{100 * l:.1f}, {100 * h:.1f}]")
    return 0


# The four Oct 7 losses: the exact 18-card pools and who drafted first ("us" = the player under test).
LIVE_POOLS: dict[str, tuple[str, list[str]]] = {
    "0a2b7c2a": ("us", ["Amoonguss", "Archaludon", "Baxcalibur", "Dragonite", "Gholdengo", "Great Tusk", "Gyarados", "Hatterene",
                        "Incineroar", "Indeedee-F", "Landorus-Therian", "Lucario", "Sableye", "Thundurus", "Ting-Lu", "Tornadus",
                        "Tyranitar", "Wo-Chien"]),
    "1aef9227": ("them", ["Archaludon", "Chien-Pao", "Cresselia", "Dondozo", "Flutter Mane", "Great Tusk", "Gyarados", "Hatterene",
                          "Indeedee-F", "Iron Bundle", "Iron Hands", "Kingambit", "Klefki", "Rillaboom", "Sylveon", "Whimsicott",
                          "Wo-Chien", "Incineroar"]),
    "7ce71025": ("us", ["Amoonguss", "Archaludon", "Dondozo", "Garchomp", "Great Tusk", "Grimmsnarl", "Hatterene", "Incineroar",
                        "Indeedee-F", "Iron Hands", "Rillaboom", "Sableye", "Sylveon", "Ting-Lu", "Tornadus", "Ursaluna",
                        "Whimsicott", "Wo-Chien"]),
    "d9405f02": ("them", ["Amoonguss", "Archaludon", "Baxcalibur", "Cresselia", "Garchomp", "Gholdengo", "Indeedee-F", "Iron Bundle",
                          "Iron Hands", "Klefki", "Landorus-Therian", "Lucario", "Rillaboom", "Sneasler", "Thundurus", "Ting-Lu",
                          "Wo-Chien", "Dragonite"]),
}


class FixedPoolRandom(random.Random):
    """A Random that hands ``harness.play_game`` a fixed pool and first drafter, then behaves normally (the battle
    seed still comes from ``seed``), so one live pool can be replayed under many battle seeds without touching the
    harness."""

    def __init__(self, seed: int, pool: list[dict], first: str) -> None:
        super().__init__(seed)
        self._pool, self._first = list(pool), first
        self._pool_given = self._first_given = False

    def sample(self, population, k, **kwargs):
        if not self._pool_given and k == len(self._pool):
            self._pool_given = True
            return list(self._pool)
        return super().sample(population, k, **kwargs)

    def choice(self, seq):
        if not self._first_given and list(seq) == ["p1", "p2"]:
            self._first_given = True
            return self._first
        return super().choice(seq)


def cmd_live(args) -> int:
    """Replay the four live pools: player A drafts from our seat (first or second, as we did live), B from theirs."""
    by_species = {c["species"]: c for c in harness.CARDS}
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    bridge = harness.Bridge()
    total_w = total_n = 0
    try:
        for match, (first, species) in LIVE_POOLS.items():
            pool = [by_species[s] for s in species]
            wins = n = 0
            rosters = None
            for g in range(args.games):
                seed = args.seed * 1_000_003 + g
                rng = FixedPoolRandom(seed, pool, "p1" if first == "us" else "p2")
                players = {"p1": make_player(args.a, seed * 2), "p2": make_player(args.b, seed * 2 + 1)}
                result = harness.play_game(bridge, f"live-{match}-{g}", players, {"p1": "A", "p2": "B"}, rng, verbose=False)
                record(out, result, seed, {"p1": args.a, "p2": args.b})
                rosters = rosters or result["rosters"]
                if result["winner"] is None:
                    continue
                n += 1
                wins += result["winner"] == "A"
            p, lo, hi = wilson(wins, n)
            total_w, total_n = total_w + wins, total_n + n
            print(f"  {match} ({'A' if first == 'us' else 'B'} drafts first): {args.a} {wins}/{n} = {100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
                  f"  A drafted {rosters['p1'] if rosters else '?'}  B drafted {rosters['p2'] if rosters else '?'}", flush=True)
    finally:
        bridge.close()
        if out:
            out.close()
    p, lo, hi = wilson(total_w, total_n)
    print(f"{args.a} vs {args.b} on the four live pools: {total_w}/{total_n} = {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
    return 0


# Team Preview on the two live matches lost at Team Preview: both sixes, their live lineup (leads first), what we
# played, and the best lineup the main session's 90-lineup sweep found against their live lineup.
LIVE_PREVIEWS: dict[str, dict] = {
    "1aef9227": {"ours": ["Iron Bundle", "Iron Hands", "Archaludon", "Gyarados", "Whimsicott", "Hatterene"],
                 "theirs": ["Incineroar", "Flutter Mane", "Sylveon", "Rillaboom", "Kingambit", "Dondozo"],
                 "their_lineup": ["Incineroar", "Sylveon", "Flutter Mane", "Kingambit"],
                 "played": ["Iron Hands", "Whimsicott", "Archaludon", "Gyarados"],
                 "sweep_best": ["Whimsicott", "Hatterene", "Archaludon", "Gyarados"]},
    "7ce71025": {"ours": ["Tornadus", "Garchomp", "Great Tusk", "Hatterene", "Ursaluna", "Indeedee-F"],
                 "theirs": ["Incineroar", "Iron Hands", "Whimsicott", "Archaludon", "Sylveon", "Amoonguss"],
                 "their_lineup": ["Iron Hands", "Sylveon", "Whimsicott", "Incineroar"],
                 "played": ["Great Tusk", "Tornadus", "Hatterene", "Indeedee-F"],
                 "sweep_best": ["Tornadus", "Garchomp", "Ursaluna", "Indeedee-F"]},
}


class _Forced:
    """Wraps a code-brain agent: drafts exactly ``roster`` and brings exactly ``lineup`` (leads first), but lets the
    agent see every draft and Team Preview state so its memory holds both full rosters for the battle."""

    def __init__(self, inner: PokemonAgent, roster: list[dict], lineup: list[str]) -> None:
        self.inner, self.roster, self.lineup = inner, roster, lineup
        self._version = getattr(inner, "_version", "forced")

    def choose_action(self, state, context):
        from agent.pokemon import data
        from agent.pokemon.memory import observation_dict

        first = state.legal_actions[0] if state.legal_actions else None
        if first is not None and first.action_id.startswith("draft_pick:"):
            self.inner.memory.observe_draft(observation_dict(state), my_turn=True, agent_id=context.agent_id)
            mine = {c["card_id"] for c in self.roster}
            pick = next(a for a in state.legal_actions if a.action_id.split(":", 1)[1] in mine)
            self.inner.memory.record_pick(pick.action_id.split(":", 1)[1])
            return pick
        if first is not None and first.action_id == "select_lineup":
            self.inner.choose_action(state, context)  # fills its memory; its own answer is not used
            ids = [data.to_id(s) for s in self.lineup]
            return {"type": "select_lineup", "bring": ids, "leads": ids[:2]}
        return self.inner.choose_action(state, context)


def preview_choices(ours: list[dict], theirs: list[dict], params: dict) -> list[str]:
    """What Team Preview (v1 or v2 per ``params``) brings, leads first, as species names."""
    from agent.pokemon import data, lineup
    from agent.pokemon.memory import species_key

    best = lineup.shortlist({species_key(c["species"]): c for c in ours}, {species_key(c["species"]): c for c in theirs},
                            [data.to_id(c["species"]) for c in ours], top=1, params=params)[0]
    by_id = {data.to_id(c["species"]): c["species"] for c in ours}
    order = list(best.leads) + [b for b in best.bring if b not in best.leads]
    return [by_id[data.to_id(b)] for b in order]


def cmd_preview(args) -> int:
    """For each live match: our candidate lineups against their live lineup, many battle seeds each."""
    by_species = {c["species"]: c for c in harness.CARDS}
    bridge = harness.Bridge()
    try:
        for match, spec in LIVE_PREVIEWS.items():
            ours = [by_species[s] for s in spec["ours"]]
            theirs = [by_species[s] for s in spec["theirs"]]
            filler = [c for c in harness.CARDS if c["species"] not in spec["ours"] + spec["theirs"]][:6]
            pool = ours + theirs + filler
            candidates = {
                "Team Preview v2": preview_choices(ours, theirs, tuning.merged({"lineup_v2": 1})),
                "old Team Preview": preview_choices(ours, theirs, tuning.merged({"lineup_v2": 0})),
                "played live": spec["played"],
                "sweep best": spec["sweep_best"],
            }
            print(f"{match}: their live lineup {spec['their_lineup']} (leads first)", flush=True)
            for name, lineup in candidates.items():
                wins = n = 0
                for g in range(args.games):
                    seed = args.seed * 1_000_003 + g
                    rng = FixedPoolRandom(seed, pool, "p1")
                    a = PokemonAgent(None, version="A", params=tuning.merged({}), log=lambda _line: None)
                    b = PokemonAgent(None, version="B", params=tuning.merged({}), log=lambda _line: None)
                    a._log_dir = b._log_dir = NO_LOG
                    players = {"p1": _Forced(a, ours, lineup), "p2": _Forced(b, theirs, spec["their_lineup"])}
                    result = harness.play_game(bridge, f"preview-{match}-{g}", players, {"p1": "A", "p2": "B"}, rng, verbose=False)
                    if result["winner"] is None:
                        continue
                    n += 1
                    wins += result["winner"] == "A"
                p, lo, hi = wilson(wins, n)
                print(f"   {name:16s} {lineup[:2]} + {lineup[2:]}: {wins}/{n} = {100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]", flush=True)
    finally:
        bridge.close()
    return 0


def cmd_collect(args) -> int:
    out = open(args.out, "a", encoding="utf-8")
    bridge = harness.Bridge()
    started = time.monotonic()
    errors = 0
    try:
        for g in range(args.games):
            seed = args.seed * 1_000_003 + g
            specs = {"p1": args.a, "p2": args.b}
            try:
                result = play(bridge, f"collect-{seed}", seed, specs, {"p1": "p1", "p2": "p2"})
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"   game {g}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            record(out, result, seed, specs)
            if args.progress and (g + 1) % args.progress == 0:
                print(f"  {g + 1} games ({int(time.monotonic() - started)} s, {errors} errors)", flush=True)
    finally:
        bridge.close()
        out.close()
    print(f"collected {args.games - errors} games in {int(time.monotonic() - started)} s ({errors} errors, "
          f"{rate(args.games - errors, started)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="paired head-to-head between two players")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--pairs", type=int, default=250, help="pools to play; each pool is played twice with seats swapped")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", default=None, help="append one JSON line per game here (optional)")
    m.add_argument("--progress", type=int, default=50)
    m.add_argument("--clock", action="store_true", help="the live clock (sim/clock.py): late decisions become random moves, an empty bank forfeits")
    m.add_argument("--log-dir", default=None, help="write the in-process agents' decision logs here (default: off)")
    m.add_argument("--pool-has", default=None, metavar="SPECIES",
                   help="play only pools holding this card; seeds without it are skipped, the same ones for every arm on one --seed")
    m.add_argument("--pin-tiebreak", action="store_true",
                   help="pin the in-process players' secret tie-break seeds per game (sha256 of battle seed and seat swap), so arms on one --seed share them")
    m.add_argument("--first-only", choices=("a", "b"), default=None,
                   help="play only the game of each pool in which player a (or b) drafts first; --pairs still counts pools")
    c = sub.add_parser("collect", help="training data: rosters and results, one JSON line per game")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--games", type=int, default=1000)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    c.add_argument("--progress", type=int, default=500)
    lv = sub.add_parser("live", help="the four Oct 7 pools, our drafter in our live seat, many battle seeds each")
    lv.add_argument("a")
    lv.add_argument("b")
    lv.add_argument("--games", type=int, default=100, help="battle seeds per pool (the drafts themselves are deterministic)")
    lv.add_argument("--seed", type=int, default=1)
    lv.add_argument("--out", default=None)
    pv = sub.add_parser("preview", help="Team Preview choices on the two live matches lost at Team Preview, vs their live lineup")
    pv.add_argument("--games", type=int, default=200, help="battle seeds per lineup")
    pv.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    return {"match": cmd_match, "collect": cmd_collect, "live": cmd_live, "preview": cmd_preview}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
