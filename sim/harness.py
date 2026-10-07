"""Local self-play: draft -> Team Preview -> full doubles battle on a real Pokémon Showdown engine.

    python sim/harness.py --games 20 --p1 code --p2 random          # free, fast
    python sim/harness.py --games 5 --p1 fable --p2 code             # real model vs the code brain
    python sim/harness.py --games 10 --p1 agent.versions.v2 --p2 agent.agent

Players: ``code`` (agent/versions/code_only.py), ``fable`` (agent/agent.py with the model chain from .env),
``random`` (random legal choices, random draft), or any ``module[:factory]`` spec. Seats swap every game so
neither side always drafts first. Each game writes the agents' decision logs under logs/sim/<player>/ and
one summary line to logs/sim/results.jsonl; the end-of-run table shows wins per player.

Requires Node and ``npm install`` in sim/ (done once).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
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


def make_player(spec: str, rng: random.Random):
    if spec == "random":
        return RandomPlayer(rng), "random"
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


def play_game(bridge: Bridge, game_id: str, players: dict[str, object], names: dict[str, str], rng: random.Random, *, verbose: bool) -> dict:
    ctx = {side: DecisionContext(session_id=game_id, tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id=side,
                                 seat_position=0 if side == "p1" else 1) for side in ("p1", "p2")}
    # ---- draft: 18-card pool, snake order, item clause
    pool = rng.sample(CARDS, 18)
    first = rng.choice(["p1", "p2"])
    order = [first if i in (0, 3, 4, 7, 8, 11) else ("p2" if first == "p1" else "p1") for i in range(12)]
    rosters: dict[str, list[dict]] = {"p1": [], "p2": []}
    picks: list[dict] = []
    version = 0
    for seat in order:
        for side in ("p1", "p2"):
            state = translate.draft_state(game_id, me=side, them="p2" if side == "p1" else "p1", pool=pool, rosters=rosters, picks=picks,
                                          first_drafter=first, current_seat=seat, version=version)
            if side != seat:
                continue  # the other seat only observes; our agent is not asked
            if not state.legal_actions:
                raise RuntimeError("item clause left no legal cards")
            decision = unwrap(players[side].choose_action(state, ctx[side]))
            card_id = decision.action_id.split(":", 1)[1] if hasattr(decision, "action_id") else decision["card_id"]
            card = next(c for c in pool if c["card_id"] == card_id)
            pool.remove(card)
            rosters[seat].append(card)
            picks.append({"pick_number": len(picks) + 1, "seat_id": seat, "card_id": card_id, "species": card["species"], "public_reason": "", "auto": False, "auto_reason": None})
            version += 1
    # ---- team preview
    lineups: dict[str, list[dict]] = {}
    for side in ("p1", "p2"):
        other = "p2" if side == "p1" else "p1"
        state = translate.preview_state(game_id, my_cards=rosters[side], opp_cards=rosters[other], version=version)
        decision = unwrap(players[side].choose_action(state, ctx[side]))
        _, ordered = translate.preview_choice(decision["bring"], decision["leads"], rosters[side])
        lineups[side] = ordered
    # ---- battle
    seed = [rng.randrange(1, 65536) for _ in range(4)]
    reply = bridge.call(cmd="new", id=game_id, format="gen9vgc2025regi", seed=seed,
                        p1={"name": names["p1"], "team": translate.team_text(lineups["p1"])},
                        p2={"name": names["p2"], "team": translate.team_text(lineups["p2"])})
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error"))
    snap = reply["state"]
    for side in ("p1", "p2"):
        reply = bridge.call(cmd="choose", id=game_id, side=side, choice="team 1234")
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error"))
        snap = reply["state"]
    versions = {"p1": 0, "p2": 0}
    rejected = {"p1": 0, "p2": 0}
    while not snap["ended"] and snap["turn"] <= MAX_TURNS:
        progressed = False
        for side in ("p1", "p2"):
            state = translate.battle_state(game_id, snap, side, versions[side])
            if state is None:
                continue
            t_dec = time.monotonic()
            decision = unwrap(players[side].choose_action(state, ctx[side]))
            if time.monotonic() - t_dec > 60:
                print(f"   [{game_id}] SLOW decision by {side}: {int(time.monotonic() - t_dec)} s", file=sys.stderr, flush=True)
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
        if not progressed:
            raise RuntimeError(f"no side had a decision at turn {snap['turn']}")
    bridge.call(cmd="close", id=game_id)
    winner_side = None
    if snap["winner"]:
        winner_side = "p1" if snap["winner"] == names["p1"] else "p2"
    return {"game": game_id, "turns": snap["turn"], "winner_side": winner_side, "winner": names.get(winner_side) if winner_side else None,
            "first_drafter": first, "rejected": rejected,
            "teams": {s: [c["species"] for c in lineups[s]] for s in ("p1", "p2")},
            "rosters": {s: [c["species"] for c in rosters[s]] for s in ("p1", "p2")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--p1", default="code")
    parser.add_argument("--p2", default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
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
                result = play_game(bridge, game_id, players, names, rng, verbose=args.verbose)
            except Exception as exc:  # noqa: BLE001
                print(f"game {g + 1}: ERROR {type(exc).__name__}: {exc}")
                continue
            if result["winner"]:
                wins[result["winner"]] = wins.get(result["winner"], 0) + 1
            print(f"game {g + 1}: {names['p1']} vs {names['p2']} -> winner {result['winner']} in {result['turns']} turns "
                  f"({int(time.monotonic() - t0)} s; rejected choices {result['rejected']})")
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
