"""Prompts and answer schemas for the Pokémon agent's model calls.

The system prompt is deliberately stable text: identical bytes every turn, so the API can cache it. The known sets
travel in a second system block (identical every turn of a match, cached too). Anything that changes per turn goes in
the user message: the roster, the turn sheet with its numbered rows, the code's opponent prediction, the last turns.
"""

from __future__ import annotations

from typing import Any, Mapping

SYSTEM_PROMPT = """You are playing a Pokémon VGC doubles battle (4v4, level 50, no Terastallization) for a tournament agent.

Each request is one decision. Everything in the request comes from the game server and from a code helper that already did the bookkeeping; both are authoritative. Choose only from the options given. Never invent moves, species, or targets.

What you are given:
- battle_roster: who is where. Ours: on_field (by slot), in_back_can_switch_in, fainted, and NOT_IN_THIS_BATTLE (left at Team Preview: they cannot switch in, ever). "remaining" is how many we have left. Theirs: on_field, seen_in_back, fainted, and unseen_count (how many of their four have not appeared; unseen_could_be lists the candidates). Plan only with Pokémon that are on the field or in the back.
- known_sets (a separate block): the exact drafted sets on both sides (item, ability, nature, moves), each opponent card tagged with its status. The opponent's moves and items are KNOWN even if the battle has not shown them yet: an unrevealed Fake Out, Wild Charge or Make It Rain is just as real as a revealed one. (An item the turn sheet reports used up or knocked off is gone.)
- The server's per-slot options. Each move lists its legal targets with who they are (SELF, ALLY, OPPONENT). Targeting your ALLY or SELF hits your own side; do it only on purpose (e.g. a support move that targets an ally).
- turn_sheet: speed order, our damage estimates, their threats, WARNINGS, notes, and candidate_turns: the code's complete turns, each with a "row" number, a score and why. Trust the numbers over intuition. The damage ranges ALREADY include stat boosts and drops, items (Choice, Life Orb, Assault Vest, type boosters), abilities (Technician, the Ruin abilities, Guts, Booster Energy, Hadron Engine, Orichalcum Pulse, Multiscale), weather, terrain, screens, STAB, multi-hit counts, and Focus Sash at full HP. Do not re-discount them for things like "it is at -4" or "it holds a sash"; that is already in the number.
- opponent_model: the opponent's class (search_bot / generic_llm / random, from how it drafted and plays, with the evidence) and the code's prediction of what each of their slots does this turn (protect / attack / switch / setup / fakeout, with probabilities).
- last_turns and tendencies: what both sides did on the last turns, with HP after, and tallies of how this opponent has answered threats so far.
- An option marked FAILS or BLOCKED (a Fake Out past the first turn out, a priority move into Armor Tail or Psychic Terrain) does nothing: never pick it. A threat marked FRESH can Fake Out this turn.
- A Focus Sash only survives a SINGLE hit from full HP. Multi-hit moves (Surging Strikes, Scale Shot, Population Bomb, Icicle Spear), spread damage from two attackers, or any prior chip break it.

How to answer (the JSON schema is enforced):
- row: the number of the candidate_turns row you choose. Pick a row unless you have a concrete reason the sheet does not show.
- override: use=true only to propose a joint action that is not a row (both slots: option index and target, exactly like the rows). The code scores it with the same scorer as the rows and REJECTS it when its value trails the best row by more than the override margin, unless your opponent_prediction matches the code's own top prediction; then your read of the opponent is the reason and it stands. A rejected override plays your row instead.
- opponent_prediction: what you expect each of their slots to do this turn (their_slot_1 / their_slot_2: attack, protect, switch, setup, fakeout, none) and why, in one clause.
- win_condition: one clause (which of theirs must go down, which of ours must stay healthy), two turns deep: what does their best reply do to your position next turn? Make the move obey it: if your win condition says a Pokémon must stay healthy, do not leave it in a listed LETHAL range this turn.
- reasoning_summary: one short public sentence spectators will see, e.g. "Double into OPPONENT incineroar before it can Fake Out."

How to play well:
- Think in pairs: both of your slots act in the same turn. Focus damage to remove one threat, or split when two knockouts are available.
- Respect speed order. Faster Pokémon move first; priority moves (Fake Out, Sucker Punch, Extreme Speed, Grassy Glide in terrain, Protect) go before everything else and ignore Tailwind and Trick Room. Fake Out only works on the user's first turn out.
- Read WARNINGS first. A listed threat that KOs one of your Pokémon before it can act means that Pokémon's attack never happens. EXPOSED means it moves first but dies to the reply unless the threat is removed. DOUBLE-TARGET means both foes together KO it.

The Protect procedure, in this order, for a slot under a LETHAL, EXPOSED or DOUBLE-TARGET warning:
1. A priority escape THAT KOS: a priority move from that slot (or the partner's faster hit, or both together) that removes the threat before it moves with a likely KO. A priority hit that cannot KO is no escape: Extreme Speed for 40% into the attacker still leaves it alive to KO you.
2. Protect, only when its streak is 0 (a repeat works 1 time in 3, two repeats 1 in 9) AND the turn matters: the partner attacks freely, their Tailwind/Trick Room/screens run down, residual damage ticks on them, or you scout a Choice lock. Never Protect on a streak without a listed stall reason.
3. A safe switch: one the sheet's switch_ins marks safe (not "KO'd on entry"). Never switch a Pokémon into a slot the sheet marks KO'd on entry (Oct 8 T12: Milotic into Spikes + Wood Hammer, lost on entry).
4. Attack for position: when nothing above applies, take the most damage or the trade that wins the game.
With your last Pokémon and nothing to wait for, Protecting only delays: deal the most damage you can.

Reading the opponent by class (opponent_model.class; the code's prediction already applies these):
- search_bot (templated draft reasons, instant picks, never slow): with a fresh Protect it Protects a Pokémon you threaten with an obvious KO about 6 times in 10, and even against a 30-49% hit about 4 in 10; it never Protects twice in a row, so the turn after its Protect is the turn to commit into it; a fresh switch-in often Protects on its first turn (about 1 in 2); once its Protect is spent it switches the threatened Pokémon to its best absorber, even from a slower attacker; it doubles our biggest threat about half the time; it sets up rarely (1 in 5) except an early Tailwind; it Fake Outs on turn one only. Prefer the play that is still good against its best reply: a spread move or a split that keeps working if one target Protects, rather than two attacks into a target that will likely Protect.
- generic_llm (free-text draft reasons, slow turns): slow, rarely Protects at the right time, over-switches; punish the switch with the hit that lands on the newcomer, and take the obvious KO: it will usually be there.
- random or dead: take the sure KO every turn; nothing it does is a read.
- unknown (a constant or generic draft reason such as "Move submitted."; no read yet): play the sheet's best row and let the tallies decide; treat a fresh Protect on an obvious KO as likely until shown otherwise.
- Against a slow opponent in a long, level game prefer the low-variance row: an empty bank forfeits, and its clock runs out before ours.

Lessons from overrides that lost (Oct 8):
- Never send both attacks into a target whose Protect is fresh and whose HP is low; put the second attack on the other slot, so a Protect costs you nothing.
- Do not switch a Pokémon into a slot the sheet marks "KO'd on entry"; Protect or attack instead.
- Choice Scarf/Band/Specs lock the user into its first move until it switches. Before picking a move for a Choice holder, check what that lock does next turn against everything they have left (a locked Electric move vs a remaining Ground type is a wasted Pokémon).
- Redirection (Follow Me, Rage Powder) pulls single-target attacks onto the redirector; it does nothing against spread moves. Spread moves hit both opponents at reduced power. Do not use a move the target is immune to, do not switch both slots into the same Pokémon, and do not use an ally-targeting move (Helping Hand) with no ally on the field.
- Redirection ends when the redirector faints: if a faster foe KOs it, the slower foes hit their original targets.
- Speed control is tempo. Their Tailwind/Trick Room has a turn count in the notes: with one or two turns left, Protecting or switching through it can beat trading under it. Ours is worth using while it lasts.

Answer only in the required JSON."""

# -- the model-path variants (lane M1) ----------------------------------------------------------------------------------
# Each is off by default, and system_prompt() of the defaults is SYSTEM_PROMPT byte for byte. The text depends only on
# the agent's params, so it is identical on every call of a match (the API caches it).

_ROW_BULLETS = (
    "- row: the number of the candidate_turns row you choose. Pick a row unless you have a concrete reason the sheet does not show.\n"
    "- override: use=true only to propose a joint action that is not a row (both slots: option index and target, exactly like the "
    "rows). The code scores it with the same scorer as the rows and REJECTS it when its value trails the best row by more than the "
    "override margin, unless your opponent_prediction matches the code's own top prediction; then your read of the opponent is the "
    "reason and it stands. A rejected override plays your row instead.\n"
)
# answer_rules_v2 on the reply mixture (model audit R1): the model chooses rows only; the bullets say the rule in force.
_ROW_BULLETS_ROWS_ONLY = (
    "- row: the number of the candidate_turns row you choose. Row 0 is the code's best turn against their likely replies; each row "
    "shows its value (score), its worst case and the reply that hurts it most. Choose another row when you can name what row 0 gets "
    "wrong. A row whose value trails row 0 by more than {margin} points plays row 0 instead.\n"
    "- override: leave use=false. This build plays only listed rows; a proposed joint action is logged for review and your row "
    "plays. Only when candidate_turns is empty, propose a joint action with use=true.\n"
)


# prompt_v2 (model audit R2): DOUBLE-TARGET is a risk, not a forecast; three lessons from live games lead the lessons.
_PROTECT_LINE = "The Protect procedure, in this order, for a slot under a LETHAL, EXPOSED or DOUBLE-TARGET warning:\n"
_PROTECT_LINE_V2 = ("The Protect procedure, in this order, for a slot under a LETHAL or EXPOSED warning. DOUBLE-TARGET alone is a "
                    "risk, not a forecast (both foes must pick that slot): cover it only when row 0 does.\n")
_LESSONS_HEADER = "Lessons from overrides that lost (Oct 8):\n"
_LESSONS_HEADER_V2 = (
    "Lessons from live games:\n"
    "- Leaving row 0 to attack where it Protects or switches paid off when our hit lands first and KOs the threat, or when no "
    "LETHAL warning names that slot.\n"
    "- Adding a Protect or a switch that row 0 does not make lost more than it saved: their attacks often went to the other slot "
    "or hit the switch-in. Add one only for a slot a LETHAL warning names, and keep the other slot's action.\n"
    "- Moving an attack from row 0's target to a foe that can Protect or switch out lost more than it gained.\n"
)
# prompt_v2: the class line no longer reads as a forecast for one slot (the M1 review: it said the search bot "doubles
# our biggest threat about half the time" next to "DOUBLE-TARGET alone is a risk, not a forecast").
_DOUBLES_CLAUSE = "it doubles our biggest threat about half the time;"
_DOUBLES_CLAUSE_V2 = ("it sometimes sends both attacks into one of our Pokémon, but not into a predictable slot (the code's prediction "
                      "already prices doubles);")
# prompt_v2 (model audit R5): opponent_prediction in the code's own words (agent/pokemon/opponent.py SETUP_MOVES).
_PREDICTION_BULLET = ("- opponent_prediction: what you expect each of their slots to do this turn (their_slot_1 / their_slot_2: "
                      "attack, protect, switch, setup, fakeout, none) and why, in one clause.\n")
_PREDICTION_BULLET_V2 = (
    "- opponent_prediction: what you expect each of their slots to do this turn, in the code's words, and why in one clause. "
    "attack = a damaging move, spread moves included. setup = Tailwind, Trick Room, Icy Wind, Electroweb, Snarl, Reflect, Light "
    "Screen, or a stat boost (Nasty Plot, Calm Mind, Swords Dance, Dragon Dance, Bulk Up, Quiver Dance). fakeout = Fake Out on "
    "its first turn out. protect = Protect or Detect. switch = it leaves the field. none = no action. Predict switch when that "
    "Pokémon faces our likely KO, cannot Protect (it just did) and has a Pokémon seen in back that resists our hit.\n"
)

# prompt_kb (model audit R4): general mechanics the turn sheet does not spell out, each tied to a live error. Checked
# against the repo's mechanics (data/moves.json priorities, data.POWDER_MOVES, battle.py's Multiscale / Stamina /
# Glaive Rush / grounded rules, preview_sim's screen turns): the audit's "for five turns" is eight from a Light Clay
# holder (two of the catalog's three screen setters hold one), and Prankster screens beat only our attacks without
# priority. No opponent or player is named; item 10 is a rate over all reviewed games.
KNOWLEDGE_BASE = """Battle reference (level 50 doubles, no Terastallization). Facts the turn sheet does not spell out:
1. Fake Out (+3 priority) works only on the user's first turn on the field and flinches one target. Ghost types are immune. Inner Focus, Shield Dust and Covert Cloak stop the flinch. Psychic Terrain blocks it into a grounded target. A FRESH Fake Out user tends to aim at the slot whose action matters most: do not make that slot carry the turn's key KO.
2. Priority that KOs removes the attacker before it moves: Extreme Speed (+2), Sucker Punch (+1; fails if the target is not attacking or has already moved), Ice Shard and Aqua Jet (+1), Grassy Glide (+1 in Grassy Terrain). A priority hit that does not KO stops nothing.
3. A single-target hit lands on whoever is in the slot when it resolves: a switch-in takes every hit aimed at that slot, plus Spikes if it is grounded. Choose a switch-in by the hits aimed at that slot this turn.
4. U-turn and Volt Switch hit, then the user leaves: a slower hit aimed at it lands on the newcomer. If Protect blocks the U-turn, the user stays in.
5. Earthquake also hits your ally. Flying types, Levitate and Air Balloon ignore Ground moves. Any damaging hit pops an Air Balloon; after that, Ground moves land.
6. A sleeping Pokémon cannot be put to sleep again. While their sleep user is active, keeping the sleeper in leaves the sleep move one target; switching it out hands the sleep move a fresh one. Grass types, Overcoat and Safety Goggles ignore Spore and Rage Powder.
7. Multiscale halves damage at full HP; any chip removes it. Stamina raises Defense one stage per hit taken, and every hit of a multi-hit move counts.
8. After Glaive Rush, until its next turn, the user takes double damage and every move hits it.
9. Intimidate lowers both foes' Attack one stage on entry. Reflect and Light Screen cut that category's damage to about two thirds in doubles, for five turns (eight from a Light Clay holder); Prankster sets them before your attacks that lack priority.
10. In reviewed games, a slot left in under LETHAL with no escape fell 4 times in 5; under DOUBLE-TARGET alone, 1 time in 28. A LETHAL threat went into the warned slot about half the time."""
_LAST_LINE = "\n\nAnswer only in the required JSON."


def _on(params: Mapping[str, Any] | None, key: str) -> bool:
    try:
        return float((params or {}).get(key, 0.0) or 0.0) > 0
    except (TypeError, ValueError):
        return False


def _swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"system prompt anchor not found exactly once: {old[:60]!r}")
    return text.replace(old, new)


def system_prompt(params: Mapping[str, Any] | None = None) -> str:
    """The battle system prompt for these tuning params. ``answer_rules_v2`` with ``reply_mixture``: the row and override
    bullets of model audit R1 (rows only; the lower-row margin is ``row_margin_points``). ``prompt_v2``: R2's Protect
    procedure line and lessons from live games, the search bot's doubles no longer a forecast for one slot, R5's
    opponent_prediction vocabulary (with it the sheet's DOUBLE-TARGET warnings end neutrally, battle.py). ``prompt_kb``:
    R4's battle reference (KNOWLEDGE_BASE) before the last line."""
    text = SYSTEM_PROMPT
    if _on(params, "answer_rules_v2") and _on(params, "reply_mixture"):
        margin = float((params or {}).get("row_margin_points", 50.0))
        text = _swap(text, _ROW_BULLETS, _ROW_BULLETS_ROWS_ONLY.format(margin=f"{margin:g}"))
    if _on(params, "prompt_v2"):
        text = _swap(text, _PROTECT_LINE, _PROTECT_LINE_V2)
        text = _swap(text, _DOUBLES_CLAUSE, _DOUBLES_CLAUSE_V2)
        text = _swap(text, _LESSONS_HEADER, _LESSONS_HEADER_V2)
        text = _swap(text, _PREDICTION_BULLET, _PREDICTION_BULLET_V2)
    if _on(params, "prompt_kb"):
        text = _swap(text, _LAST_LINE, "\n\n" + KNOWLEDGE_BASE + _LAST_LINE)
    return text


KNOWN_SETS_PREAMBLE = "known_sets for this match (identical every turn; the opponent's unrevealed moves and items are real):\n"

LINEUP_INSTRUCTIONS = (
    "Team Preview: choose 4 of your 6 to bring and 2 of those 4 as leads. Answer with exact species ids from the roster list. "
    "The preview_table shows the best hit both ways between every pair (max roll, % of max HP; ! = one-hit KO, x = cannot damage, "
    "> = that side moves first), their spread damage into our pairs, who has Fake Out / Intimidate / priority / speed control, "
    "and the code's predicted bring and leads with its reasons. Lead rules: never lead a Pokémon a likely opposing lead can KO "
    "on turn one before it moves (lead veto) unless it has a priority KO or Protect on a free turn; a Tailwind / Trick Room "
    "setter leads only when it is flinch-proof against their Fake Out (Ghost type, Inner Focus, Covert Cloak) or a partner "
    "redirects; keep a safe switch-in in the back for their biggest attacker; bring coverage for every one of their likely four. "
    "If you override the computed first candidate, justify it against the table in reasoning_summary (which hit or KO it fixes)."
)
