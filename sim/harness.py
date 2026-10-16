"""Local self-play: draft -> Team Preview -> full doubles battle on a real Pokémon Showdown engine.

    python sim/harness.py --games 20 --p1 code --p2 random          # free, fast
    python sim/harness.py --games 5 --p1 fable --p2 code             # real model vs the code brain
    python sim/harness.py --games 10 --p1 agent.versions.v2 --p2 agent.agent

Players: ``code`` (agent/versions/code_only.py), ``fable`` (agent/agent.py with the model chain from .env),
``random`` (random legal choices, random draft), or any ``module[:factory]`` spec. Seats swap every game so
neither side always drafts first. Each game writes the agents' decision logs under logs/sim/<player>/ and
one summary line to logs/sim/results.jsonl; the end-of-run table shows wins per player. ``--clock`` enforces the
live clock (sim/clock.py): a late Team Preview or battle decision becomes a random legal move, an empty bank forfeits,
and every result line carries the clock events.

Oracle hook (sim only, opt-in): a player with ``wants_opponent_view = True`` is shown the other seat's state for every
decision the other seat makes, through its ``observe_opponent(state, context)``, before its own decision of the same
step (``oracle_of``); agent/versions/reader.py reads our play that way. Other players never see it and play as before.

Requires Node and ``npm install`` in sim/ (done once).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from altruagent import DecisionContext, WithReasoning  # noqa: E402
from altruagent.agent_loader import load_agent_factory  # noqa: E402

from agent.pokemon import data  # noqa: E402
from sim import translate  # noqa: E402
from sim.clock import Clock  # noqa: E402

CARDS = json.loads((ROOT / "sim" / "cards.json").read_text())
MAX_TURNS = 120


class Bridge:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(["node", str(ROOT / "sim" / "bridge.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, **msg) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("bridge died")
        return json.loads(line)

    def close(self) -> None:
        self.proc.terminate()


class RandomPlayer:
    """Baseline: random legal choices everywhere."""

    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_action(self, state, context):
        from examples.llm.pokemon import doubles_choice, lineup_choice

        first = state.legal_actions[0]
        if first.action_id.startswith("draft_pick:"):
            return self.rng.choice(state.legal_actions)
        if first.action_id == "select_lineup":
            roster = list(first.input["action"]["roster"])
            bring = self.rng.sample(roster, 4)
            return {"type": "select_lineup", "bring": bring, "leads": bring[:2]}
        template = first.input["action"]
        for _ in range(20):
            answer = {}
            for s in template["slots"]:
                opts = s["options"]
                i = self.rng.randrange(len(opts))
                targets = opts[i].get("targets") or []
                answer[f"slot_{s['slot']}"] = {"option": i, "target": self.rng.choice(targets) if targets else 0}
            try:
                return doubles_choice(first, state).build(answer)
            except Exception:
                continue
        from examples import smoke_agent

        return smoke_agent.choose_action(state, context)


class RestartingPlayer:
    """Reliability drill: ``<spec>@restartN`` throws the agent away at battle turn N and builds a fresh one from
    ``create_agent()``, the way the runtime does when a game's process dies and is started again. The fresh agent has
    no draft or Team Preview memory and must recover everything from the battle observation."""

    def __init__(self, spec: str, restart_turn: int, rng: random.Random):
        self.spec, self.restart_turn, self.rng = spec, restart_turn, rng
        self.inner, self.name = make_player(spec, rng)
        self.restarted = False

    @property
    def _version(self):
        return getattr(self.inner, "_version", self.name)

    def choose_action(self, state, context):
        from agent.pokemon.memory import observation_dict

        obs = observation_dict(state)
        if not self.restarted and state.phase == "moving" and int(obs.get("turn") or 0) >= self.restart_turn:
            self.inner, _ = make_player(self.spec, self.rng)
            self.restarted = True
            print(f"   [restart drill] fresh agent at turn {obs.get('turn')} (no draft/preview memory)", file=sys.stderr, flush=True)
        return self.inner.choose_action(state, context)


def make_player(spec: str, rng: random.Random):
    if "@restart" in spec:
        base, _, n = spec.partition("@restart")
        return RestartingPlayer(base, int(n), rng), f"{base}@restart{n}"
    if spec.startswith("checkout:"):  # an agent from another checkout (e.g. the Oct 7 version), in a child process
        from sim.subprocess_player import make

        return make(spec), spec
    if spec == "random":
        return RandomPlayer(rng), "random"
    if spec == "human":
        from sim.human import HumanPlayer

        return HumanPlayer(), "human"
    if spec == "code":
        spec = "agent.versions.code_only"
    elif spec == "fable":
        spec = "agent.agent"
    agent = load_agent_factory(spec)()
    if not hasattr(agent, "choose_action"):  # a bare decision function (e.g. examples.smoke_agent)
        fn = agent

        class _Fn:
            def choose_action(self, state, context):
                return fn(state, context)

        agent = _Fn()
    return agent, spec


def unwrap(decision):
    return decision.action if isinstance(decision, WithReasoning) else decision


def timed_decision(player, state, context, game_id: str, side: str, phase: str, clock: Clock | None, rng: random.Random, *, turn: int):
    """One decision with its wall time measured. Under the live clock a late answer becomes a random legal move and an
    empty bank forfeits; the verdict is "ok", "timeout" or "forfeit"."""
    now = clock.now if clock else time.monotonic
    t0 = now()
    decision = unwrap(player.choose_action(state, context))
    elapsed = now() - t0
    if elapsed > 60:
        print(f"   [{game_id}] SLOW decision by {side}: {int(elapsed)} s", file=sys.stderr, flush=True)
    if clock is None:
        return decision, "ok"
    verdict = clock.settle(side, phase, elapsed, turn=turn)
    if verdict == "timeout":
        print(f"   [{game_id}] CLOCK: {side} took {elapsed:.1f} s at {phase} turn {turn}; a random legal move is played", file=sys.stderr, flush=True)
        decision = unwrap(RandomPlayer(rng).choose_action(state, context))
    elif verdict == "forfeit":
        print(f"   [{game_id}] CLOCK: {side} forfeits on an empty bank at {phase} turn {turn}", file=sys.stderr, flush=True)
    return decision, verdict


def seat_ids(game_id: str) -> dict[str, str]:
    """Agent ids for the two seats. Live, seats, rosters, picks and current_actor are keyed by the agents' UUIDs;
    deterministic per game so seeded runs stay reproducible (and no draw from the game's rng)."""
    return {side: str(uuid.uuid5(uuid.NAMESPACE_URL, f"ace-trainer-sim/{game_id}/{side}")) for side in ("p1", "p2")}


def oracle_of(player):
    """The oracle hook of a player that asks for it (sim-only, opt-in): its ``observe_opponent(state, context)`` when it
    (or a wrapper's ``inner`` player) sets ``wants_opponent_view = True``, else None. Such a player is shown the other
    seat's state for every decision the other seat makes (draft, Team Preview, every battle decision), before its own
    decision of the same step: a practice opponent that reads our play (agent/versions/reader.py). Players without the
    attribute are never shown anything and play exactly as before."""
    for _ in range(4):  # through _Timed / RestartingPlayer style wrappers
        if player is None:
            return None
        if getattr(player, "wants_opponent_view", False) is True:
            hook = getattr(player, "observe_opponent", None)
            return hook if callable(hook) else None
        player = getattr(player, "inner", None)
    return None


def request_number(snap: dict, side: str) -> int:
    """How many requests the engine has handed ``side`` so far (sim/bridge.js logs one battle-timer line per new
    request): tells a seat's new request from the one it already answered, which stays in the snapshot until the turn
    moves on. Oracle hook only."""
    return sum(1 for line in (snap.get("logs") or {}).get(side) or [] if line.startswith("|inactive|Time left"))


def _show(hook, state, context, game_id: str) -> None:
    """Hand the other seat's state to an oracle hook; a failing hook costs nothing but a console line."""
    try:
        hook(state, context)
    except Exception as exc:  # noqa: BLE001 — the reader's own fallback plays its turn without the view
        print(f"   [{game_id}] oracle hook failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)


def play_game(bridge: Bridge, game_id: str, players: dict[str, object], names: dict[str, str], rng: random.Random, *, verbose: bool,
              fixed_rosters: dict[str, list[dict]] | None = None, clock: Clock | None = None) -> dict:
    """One match the way the live server runs it: draft (seat 0 is Showdown's p1), Team Preview over all six drafted
    Pokémon, then the doubles battle. Every state the agents see is built by sim/translate.py in the live format.
    ``fixed_rosters`` ({"p1": six cards, "p2": six cards}) skips the draft: replay a real match's teams to ask how
    winnable the draft left it (scripts/draft_eval.py). ``clock`` (sim/clock.py) times Team Preview and battle decisions
    the way the live server does: late answers become random legal moves, an empty bank forfeits."""
    seats = seat_ids(game_id)
    ctx = {side: DecisionContext(session_id=game_id, tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id=seats[side],
                                 seat_position=0 if side == "p1" else 1) for side in ("p1", "p2")}
    # ---- draft: 18-card pool, snake order, item clause (only the seat to pick is asked, as live)
    pool = rng.sample(CARDS, 18)
    first = rng.choice(["p1", "p2"])
    order = [first if i in (0, 3, 4, 7, 8, 11) else ("p2" if first == "p1" else "p1") for i in range(12)]
    rosters: dict[str, list[dict]] = {seats["p1"]: [], seats["p2"]: []}  # seat 0 first, as live
    picks: list[dict] = []
    pick_log: list[dict] = []  # per pick: side, species, the public reason (what the live draft shows the other seat)
    version = 0
    if fixed_rosters:
        rosters = {seats["p1"]: list(fixed_rosters["p1"]), seats["p2"]: list(fixed_rosters["p2"])}
        order = []
    for side in order:
        state = translate.draft_state(game_id, me=seats[side], pool=pool, rosters=rosters, picks=picks,
                                      first_drafter=seats[first], current_seat=seats[side], version=version)
        if not state.legal_actions:
            raise RuntimeError("item clause left no legal cards")
        other = "p2" if side == "p1" else "p1"
        hook = oracle_of(players[other])
        shown = copy.deepcopy(state) if hook else None  # what this seat sees, untouched by its own decision
        answer = players[side].choose_action(state, ctx[side])
        if hook:
            _show(hook, shown, ctx[side], game_id)
        decision = unwrap(answer)
        card_id = decision.action_id.split(":", 1)[1] if hasattr(decision, "action_id") else decision["card_id"]
        card = next(c for c in pool if c["card_id"] == card_id)
        pool.remove(card)
        rosters[seats[side]].append(card)
        reason = answer.reasoning_summary if isinstance(answer, WithReasoning) else None
        picks.append({"pick_number": len(picks) + 1, "seat_id": seats[side], "card_id": card_id, "species": card["species"],
                      "public_reason": reason or "", "auto": False, "auto_reason": None})
        pick_log.append({"side": side, "species": card["species"], "public_reason": reason or ""})
        version += 1
    # ---- team preview: Showdown gets all six (draft order) and each side picks 4, leads first
    seed = [rng.randrange(1, 65536) for _ in range(4)]
    showdown_names = {"p1": f"{names['p1']}-p0", "p2": f"{names['p2']}-p1"}
    reply = bridge.call(cmd="new", id=game_id, format="gen9vgc2025regi", seed=seed,
                        p1={"name": showdown_names["p1"], "team": translate.team_text(rosters[seats["p1"]])},
                        p2={"name": showdown_names["p2"], "team": translate.team_text(rosters[seats["p2"]])})
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error"))
    snap = reply["state"]
    choices: dict[str, str] = {}
    lineups: dict[str, list[int]] = {}
    previews: dict[str, object] = {}  # oracle hook only: each seat's Team Preview state, as it saw it
    forfeit: str | None = None
    for side in ("p1", "p2"):
        state = translate.preview_state(game_id, snap, side, version=0, agent_id=seats[side],
                                        clock=clock.state(side, "team_preview") if clock else None)
        if state is None:
            raise RuntimeError(f"no Team Preview request for {side}")
        hook = oracle_of(players[side])
        if hook:  # the oracle hook: the other seat's Team Preview state first (built from the same snapshot)
            other = "p2" if side == "p1" else "p1"
            seen = previews.get(other) or translate.preview_state(game_id, snap, other, version=0, agent_id=seats[other],
                                                                  clock=clock.state(other, "team_preview") if clock else None)
            if seen is not None:
                _show(hook, copy.deepcopy(seen), ctx[other], game_id)
        if oracle_of(players["p2" if side == "p1" else "p1"]):
            previews[side] = copy.deepcopy(state)
        decision, verdict = timed_decision(players[side], state, ctx[side], game_id, side, "team_preview", clock, rng, turn=0)
        if verdict == "forfeit":
            forfeit = side
            break
        roster = state.legal_actions[0].input["action"]["roster"]
        choices[side], lineups[side] = translate.preview_choice(decision["bring"], decision["leads"], roster)
    if forfeit is None:
        for side in ("p1", "p2"):
            reply = bridge.call(cmd="choose", id=game_id, side=side, choice=choices[side])
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error"))
            snap = reply["state"]
    # ---- battle (state_version counts each seat's decisions; Team Preview was 0)
    versions = {"p1": 1, "p2": 1}
    rejected = {"p1": 0, "p2": 0}
    shown_request: dict[str, int | None] = {"p1": None, "p2": None}  # oracle hook: each seat's request last shown to its reader
    decided_request: dict[str, int | None] = {"p1": None, "p2": None}  # ... and the one it last decided (a watched seat only)
    while forfeit is None and not snap["ended"] and snap["turn"] <= MAX_TURNS:
        progressed = False
        for side in ("p1", "p2"):
            state = translate.battle_state(game_id, snap, side, versions[side], agent_id=seats[side],
                                           clock=clock.state(side, "moving") if clock else None)
            if state is None:
                continue
            other = "p2" if side == "p1" else "p1"
            reads, watched_by = oracle_of(players[side]), oracle_of(players[other])  # oracle hooks (None for normal players)
            if reads:
                # this seat reads the other: the other seat's pending request (not yet decided, not yet shown) first,
                # built from the same snapshot (a seat's choice does not change the other seat's request)
                pending = request_number(snap, other)
                if pending not in (shown_request[other], decided_request[other]):
                    seen = translate.battle_state(game_id, snap, other, versions[other], agent_id=seats[other],
                                                  clock=clock.state(other, "moving") if clock else None)
                    if seen is not None:
                        _show(reads, seen, ctx[other], game_id)
                        shown_request[other] = pending
            mine = request_number(snap, side) if watched_by else None
            shown = copy.deepcopy(state) if watched_by and shown_request[side] != mine else None
            decision, verdict = timed_decision(players[side], state, ctx[side], game_id, side, "moving", clock, rng, turn=snap["turn"])
            if watched_by:
                decided_request[side] = mine
            if shown is not None:  # the other seat reads this one and was not shown this request yet: shown now
                _show(watched_by, shown, ctx[side], game_id)
                shown_request[side] = mine
            if verdict == "forfeit":
                forfeit = side
                break
            choice = translate.to_choice(decision)
            reply = bridge.call(cmd="choose", id=game_id, side=side, choice=choice)
            if not reply.get("ok"):
                rejected[side] += 1
                if verbose:
                    print(f"   [{game_id}] {side} choice rejected: {choice} :: {reply.get('error', '')[:160]}")
                # Fall back to the first legal thing the engine will take.
                from examples import smoke_agent

                fallback = translate.to_choice(unwrap(smoke_agent.choose_action(state, ctx[side])))
                reply = bridge.call(cmd="choose", id=game_id, side=side, choice=fallback)
                if not reply.get("ok"):
                    reply = bridge.call(cmd="choose", id=game_id, side=side, choice="default")
                if not reply.get("ok"):
                    raise RuntimeError(f"{side} stuck: {reply.get('error')}")
            snap = reply["state"]
            versions[side] += 1
            progressed = True
        if forfeit is None and not progressed:
            raise RuntimeError(f"no side had a decision at turn {snap['turn']}")
    bridge.call(cmd="close", id=game_id)
    winner_side = None
    if forfeit:
        winner_side = "p2" if forfeit == "p1" else "p1"
    elif snap["winner"]:
        winner_side = "p1" if snap["winner"] == showdown_names["p1"] else "p2"
    return {"game": game_id, "turns": snap["turn"], "winner_side": winner_side, "winner": names.get(winner_side) if winner_side else None,
            "first_drafter": first, "rejected": rejected, "forfeit": forfeit, "clock": clock.summary() if clock else None,
            "versions": {s: getattr(players[s], "_version", getattr(players[s], "name", names[s])) for s in ("p1", "p2")},
            "teams": {s: [rosters[seats[s]][i]["species"] for i in lineups.get(s, [])] for s in ("p1", "p2")},
            "rosters": {s: [c["species"] for c in rosters[seats[s]]] for s in ("p1", "p2")}, "picks": pick_log}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--p1", default="code", help="code | fable (the default agent, Opus) | random | human | a module path")
    parser.add_argument("--p2", default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--clock", action="store_true", help="the live clock: 85 s / 50 s per decision from a 420 s bank; late = random move, empty = forfeit")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    out_dir = ROOT / "logs" / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge = Bridge()
    wins: dict[str, int] = {}
    names_seen: list[str] = []
    started = time.monotonic()
    try:
        for g in range(args.games):
            specs = {"p1": args.p1, "p2": args.p2} if g % 2 == 0 else {"p1": args.p2, "p2": args.p1}
            players, names = {}, {}
            for side, spec in specs.items():
                log_dir = out_dir / spec.replace(":", "_").replace("/", "_")
                os.environ["AGENT_LOG_DIR"] = str(log_dir)
                players[side], names[side] = make_player(spec, rng)
                if hasattr(players[side], "_log_dir"):  # PokemonAgent: pin the directory now, not at first decision
                    players[side]._log_dir = str(log_dir)
                names[side] = spec  # keep the CLI spelling as the display name
            for n in names.values():
                if n not in names_seen:
                    names_seen.append(n)
            game_id = f"sim-{int(time.time())}-{g}"
            t0 = time.monotonic()
            try:
                result = play_game(bridge, game_id, players, names, rng, verbose=args.verbose, clock=Clock() if args.clock else None)
            except Exception as exc:  # noqa: BLE001
                print(f"game {g + 1}: ERROR {type(exc).__name__}: {exc}")
                continue
            result["seconds"] = round(time.monotonic() - t0, 3)
            if result["winner"]:
                wins[result["winner"]] = wins.get(result["winner"], 0) + 1
            print(f"game {g + 1}: {names['p1']} vs {names['p2']} -> winner {result['winner']} in {result['turns']} turns "
                  f"({result['seconds']:.1f} s; rejected choices {result['rejected']})")
            if result.get("clock"):
                events = [f"{side} {e['kind']} at {e['phase']} turn {e['turn']} ({e['seconds']} s)" for side, c in result["clock"].items() for e in c["events"]]
                print(f"   clock: bank left p1 {result['clock']['p1']['bank_left']} s, p2 {result['clock']['p2']['bank_left']} s"
                      + (f"; {'; '.join(events)}" if events else ""))
            with (out_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "p1": names["p1"], "p2": names["p2"], **result}) + "\n")
    finally:
        bridge.close()
    total = sum(wins.values())
    print(f"\n{total} decided games in {int(time.monotonic() - started)} s")
    for n in names_seen:
        print(f"  {n:28s} {wins.get(n, 0):3d} wins  ({100 * wins.get(n, 0) / total if total else 0:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
