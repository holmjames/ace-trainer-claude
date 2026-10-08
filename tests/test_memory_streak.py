"""Protect streaks: the memory counts consecutive Protect-like moves by the Pokémon still standing in a slot."""

from agent.pokemon.memory import MatchMemory


def _turn(mem, slot0, slot1, actives):
    mem.last_active = list(actives)
    mem.record_turn(turn=len(mem.turns) + 1, payload={"type": "doubles_turn", "slot_0": slot0, "slot_1": slot1}, model=None)


def test_protect_streak_counts_consecutive_protects_for_the_same_pokemon():
    mem = MatchMemory()
    attack = {"type": "move", "move_id": "moonblast", "target": 1}
    protect = {"type": "move", "move_id": "protect"}
    assert mem.protect_streak("hatterene", 0) == 0
    _turn(mem, protect, attack, ["hatterene", "zamazentacrowned"])
    assert mem.protect_streak("hatterene", 0) == 1 and mem.protect_streak("zamazentacrowned", 1) == 0
    _turn(mem, protect, attack, ["hatterene", "zamazentacrowned"])
    assert mem.protect_streak("hatterene", 0) == 2
    _turn(mem, attack, protect, ["hatterene", "zamazentacrowned"])
    assert mem.protect_streak("hatterene", 0) == 0 and mem.protect_streak("zamazentacrowned", 1) == 1


def test_protect_streak_resets_when_a_different_pokemon_takes_the_slot():
    mem = MatchMemory()
    protect = {"type": "move", "move_id": "detect"}
    _turn(mem, protect, {"type": "move", "move_id": "knockoff", "target": 2}, ["garchomp", "incineroar"])
    assert mem.protect_streak("garchomp", 0) == 1
    assert mem.protect_streak("lunala", 0) == 0  # Lunala just switched in: its Protect is fresh
