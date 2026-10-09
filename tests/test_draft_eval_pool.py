"""sim/draft_eval.py match --pool-has: the pool a battle seed gives (draft_eval.pool_of, harness.play_game's own draws)
and the deterministic seed filter, so both arms of an A/B on one --seed play the same pools in the same seats."""

from __future__ import annotations

import random

import pytest

from sim import draft_eval, harness, translate


class _StopAfterDraft(Exception):
    pass


class _NoEngine:
    """A bridge that ends the game where the engine would start (after the draft)."""

    def call(self, **msg):
        raise _StopAfterDraft(msg.get("cmd"))

    def close(self) -> None:
        pass


class _FirstLegal:
    def choose_action(self, state, context):
        return state.legal_actions[0]


def _has(seed: int, species: str) -> bool:
    return any(c["species"] == species for c in draft_eval.pool_of(seed)[0])


def test_pool_of_is_the_pool_and_first_drafter_play_game_draws(monkeypatch):
    seen: list[tuple[list[str], str]] = []
    original = translate.draft_state

    def spy(*args, **kwargs):
        seen.append(([c["card_id"] for c in kwargs["pool"]], kwargs["first_drafter"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(translate, "draft_state", spy)
    for seed in (7_000_021, 7_000_022, 5_000_015):
        seen.clear()
        with pytest.raises(_StopAfterDraft):
            harness.play_game(_NoEngine(), f"pool-{seed}", {"p1": _FirstLegal(), "p2": _FirstLegal()}, {"p1": "A", "p2": "B"},
                              random.Random(seed), verbose=False)
        pool, first = draft_eval.pool_of(seed)
        assert len(seen) == 12 and seen[0][0] == [c["card_id"] for c in pool]
        assert seen[0][1] == harness.seat_ids(f"pool-{seed}")[first]


def test_match_seeds_skip_pools_without_the_species_the_same_way_every_time():
    assert draft_eval.match_seeds(7, 4) == [7_000_021, 7_000_022, 7_000_023, 7_000_024]  # no filter: unchanged
    seeds = draft_eval.match_seeds(7, 6, "Archaludon")
    assert len(seeds) == 6 and seeds == draft_eval.match_seeds(7, 6, "archaludon")
    assert all(_has(s, "Archaludon") for s in seeds)
    skipped = [s for s in range(7_000_021, seeds[-1]) if s not in seeds]
    assert skipped and not any(_has(s, "Archaludon") for s in skipped)
    assert draft_eval.match_seeds(7, 3, "Archaludon") == seeds[:3]  # more pairs extend the list, never reorder it
    with pytest.raises(SystemExit):
        draft_eval.match_seeds(7, 1, "Missingno")


def test_match_with_pool_has_plays_those_pools_in_both_seats_for_every_arm(monkeypatch):
    played: list[tuple] = []

    def fake_play(bridge, game_id, seed, specs, names, *, clock=False):
        first = draft_eval.pool_of(seed)[1]
        played.append((seed, game_id, dict(specs), first))
        return {"winner": "A", "winner_side": "p1", "first_drafter": first, "turns": 1, "clock": None}

    monkeypatch.setattr(draft_eval, "play", fake_play)
    monkeypatch.setattr(harness, "Bridge", _NoEngine)
    assert draft_eval.main(["match", "v2l", "v1", "--pairs", "3", "--seed", "7", "--pool-has", "Archaludon"]) == 0
    seeds = draft_eval.match_seeds(7, 3, "Archaludon")
    assert [p[0] for p in played] == [s for s in seeds for _ in (0, 1)]
    assert [p[2] for p in played[:2]] == [{"p1": "v2l", "p2": "v1"}, {"p1": "v1", "p2": "v2l"}]
    control = [(p[0], p[1], p[3]) for p in played]
    played.clear()
    assert draft_eval.main(["match", 'v2l{"draft_first_adjust_w": 2}', "v1", "--pairs", "3", "--seed", "7", "--pool-has", "Archaludon"]) == 0
    assert [(p[0], p[1], p[3]) for p in played] == control  # same pools, battle seeds, game ids and first drafters
