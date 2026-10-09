"""sim/draft_eval.py match --pin-tiebreak and --first-only (harness only; live never loads them).

--pin-tiebreak: per game, the in-process players' secret tie-break seed is fixed from sha256(f"{battle seed}:{swap}")
(draft._SEEDS cleared, draft.secrets.randbits patched, the real one restored afterwards), so two arms run on one
--seed make the same draft and Team Preview draws game by game. --first-only a|b: only the game of each pool in which
that player drafts first; --pairs still counts pools. Without the flags nothing changes."""

from __future__ import annotations

import hashlib

import pytest

from agent.pokemon import draft
from sim import draft_eval, harness


class _StopAfterDraft(Exception):
    pass


class _NoEngine:
    """A bridge that ends the game where the engine would start (after the draft)."""

    def call(self, **msg):
        raise _StopAfterDraft(msg.get("cmd"))

    def close(self) -> None:
        pass


def _sha(seed: int, swap: bool) -> int:
    return int(hashlib.sha256(f"{seed}:{swap}".encode()).hexdigest()[:16], 16)


def _result(seed: int) -> dict:
    return {"winner": "A", "winner_side": "p1", "first_drafter": draft_eval.pool_of(seed)[1], "turns": 1, "clock": None}


@pytest.fixture
def stub_engine(monkeypatch):
    monkeypatch.setattr(harness, "Bridge", _NoEngine)


def test_pinned_seed_is_sha256_of_the_battle_seed_and_the_seat_swap():
    assert draft_eval.pinned_seed(7_000_021, False) == _sha(7_000_021, False)
    assert draft_eval.pinned_seed(7_000_021, True) == _sha(7_000_021, True) != _sha(7_000_021, False)
    assert draft_eval.pinned_seed(7_000_021, True).bit_length() <= 64


def test_without_the_flags_every_game_is_played_with_the_real_draw(monkeypatch, stub_engine):
    real = draft.secrets.randbits
    monkeypatch.setitem(draft._SEEDS, 424242, 9)  # a registry entry nobody may clear without --pin-tiebreak
    played: list[tuple] = []

    def fake_play(bridge, game_id, seed, specs, names, *, clock=False):
        played.append((seed, game_id, dict(specs), draft.secrets.randbits is real, draft._SEEDS.get(424242)))
        return _result(seed)

    monkeypatch.setattr(draft_eval, "play", fake_play)
    assert draft_eval.main(["match", "v2l", "v1", "--pairs", "3", "--seed", "7"]) == 0
    assert draft_eval.match_seeds(7, 3) == [7_000_021, 7_000_022, 7_000_023]  # the seed list is unchanged
    assert [(p[0], p[1]) for p in played] == [(s, f"eval-{s}-{w}") for s in (7_000_021, 7_000_022, 7_000_023) for w in (0, 1)]
    assert [p[2] for p in played[:2]] == [{"p1": "v2l", "p2": "v1"}, {"p1": "v1", "p2": "v2l"}]
    assert all(p[3] and p[4] == 9 for p in played)


def test_pin_tiebreak_fixes_each_games_seed_and_restores_the_real_draw(monkeypatch, stub_engine):
    real = draft.secrets.randbits
    seen: list[tuple] = []

    def fake_play(bridge, game_id, seed, specs, names, *, clock=False):
        seen.append((seed, game_id.endswith("-1"), dict(draft._SEEDS), draft.secrets.randbits(64), draft.secrets.randbits(8)))
        draft._SEEDS[12345] = 1  # a stale registry entry the next game must not see
        if len(seen) == 2:
            raise RuntimeError("a broken game")  # the real draw is restored on errors too, and the run goes on
        return _result(seed)

    monkeypatch.setattr(draft_eval, "play", fake_play)
    assert draft_eval.main(["match", "v2l", "v1", "--pairs", "2", "--seed", "7", "--pin-tiebreak"]) == 0
    assert [(s[0], s[1]) for s in seen] == [(s, w) for s in draft_eval.match_seeds(7, 2) for w in (False, True)]
    for seed, swap, registry, bits64, bits8 in seen:
        assert registry == {} and bits64 == _sha(seed, swap) and bits8 == _sha(seed, swap) & 0xFF
    assert len({s[3] for s in seen}) == 4  # each game its own seed
    assert draft.secrets.randbits is real and draft._SEEDS == {}


def test_first_only_plays_one_game_per_pool_the_one_that_player_drafts_first(monkeypatch, stub_engine):
    seen: list[tuple] = []

    def fake_play(bridge, game_id, seed, specs, names, *, clock=False):
        seen.append((seed, game_id.endswith("-1"), dict(specs), draft_eval.pool_of(seed)[1], draft.secrets.randbits(64)))
        return _result(seed)

    monkeypatch.setattr(draft_eval, "play", fake_play)
    pools = draft_eval.match_seeds(7, 6, "Archaludon")
    for who, extra in (("a", []), ("b", []), ("b", ["--pin-tiebreak"])):
        seen.clear()
        argv = ["match", "v2l", "v1", "--pairs", "6", "--seed", "7", "--pool-has", "Archaludon", "--first-only", who, *extra]
        assert draft_eval.main(argv) == 0
        assert [s[0] for s in seen] == pools  # one game per pool, the pools --pool-has picks
        for seed, swap, specs, first, bits in seen:
            drafter = specs[first]  # the spec seated where the first drafter sits
            assert drafter == ("v2l" if who == "a" else "v1")
            if extra:
                assert bits == _sha(seed, swap)
        assert {s[1] for s in seen} == {False, True}  # the seat order varies with the pool's first drafter


def test_two_pinned_runs_draft_the_same_picks_with_the_pinned_seed(monkeypatch, stub_engine):
    """The real drafter in-process, the engine stubbed after the draft: same picks and the same seed in both runs."""
    original = draft.choose_pick

    def run() -> list[tuple[str, str]]:
        picks: list[tuple[str, str]] = []

        def spy(legal_actions, memory, params=None):
            pick = original(legal_actions, memory, params)
            seed_note = next((n for n in pick.notes if n.startswith("tie-break seed")), "")
            picks.append((pick.species, seed_note.split(":")[0]))
            return pick

        monkeypatch.setattr(draft, "choose_pick", spy)
        assert draft_eval.main(["match", "v2l", "v1", "--pairs", "1", "--seed", "7", "--pin-tiebreak"]) == 0
        return picks

    first, second = run(), run()
    assert len(first) == 24 and first == second  # the pool's two games, twelve picks each: v2l's six (seeded) and v1's six
    seeds = {note for _, note in first if note}
    assert seeds == {f"tie-break seed {_sha(7_000_021, False):#x}", f"tie-break seed {_sha(7_000_021, True):#x}"}
