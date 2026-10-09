# Card tiers (live catalog)

Built by `scripts/build_tiers.py` from the fitted team model; the drafter's base value per card. Rating = average worth on a team in 1/100 log-odds of winning (sim rating + a small prior where the code-only simulator is known to misplay the card). Rebuild after refitting `data/draft_model.json`. Iron Valiant, Iron Treads, Talonflame, Toxapex, Farigiraf, Chi-Yu and Urshifu-Rapid-Strike (first seen live Oct 8) carry their matchup part only.

| Tier | Card | Rating | Sim | Why | Flag / live |
|---|---|---:|---:|---|---|
| S | Archaludon | +61 | +61 | wins 27/40 1-on-1s (best: Gyarados, Kingambit); countered by Gholdengo, Flutter Mane; wins more in simulated games than its matchups explain | the Oct 7 opponent took it 3/4 |
| S | Gholdengo | +53 | +53 | wins 18/40 1-on-1s (best: Archaludon, Porygon2); countered by Torkoal, Flutter Mane; spread; wins more in simulated games than its matchups explain | the Oct 7 opponent took it 2/2 |
| A | Torkoal | +25 | +25 | wins 16/40 1-on-1s (best: Gholdengo, Kingambit); countered by Flutter Mane, Iron Hands; spread, sleep move; wins more in simulated games than its matchups explain |  |
| A | Flutter Mane | +23 | +23 | wins 20/40 1-on-1s (best: Archaludon, Gholdengo); countered by Iron Hands, Gyarados; spread; wins more in simulated games than its matchups explain |  |
| A | Iron Hands | +21 | +21 | wins 23/40 1-on-1s (best: Archaludon, Gholdengo); countered by Sylveon, Garchomp; Fake Out | the Oct 7 opponent took it 2/3 |
| A | Gyarados | +18 | +18 | wins 26/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Iron Hands; Intimidate, spread |  |
| A | Porygon2 | +16 | +16 | wins 16/40 1-on-1s (best: Torkoal, Tornadus); countered by Gholdengo, Iron Hands; Trick Room | engine sets Trick Room whether or not its own team wants it |
| A | Kingambit | +16 | +16 | wins 22/40 1-on-1s (best: Gholdengo, Flutter Mane); countered by Archaludon, Torkoal; priority |  |
| A | Chi-Yu | +16 | +16 | wins 28/47 1-on-1s (best: Archaludon, Gholdengo); countered by Iron Hands, Gyarados; spread; not in the fitted model (first seen live Oct 8): matchup part only | not in the fitted model (first seen live Oct 8); Beads of Ruin (Sp. Def x0.75 for every other Pokémon), which the damage math does credit |
| A | Milotic | +14 | +14 | wins 13/40 1-on-1s (best: Torkoal, Incineroar); countered by Archaludon, Iron Hands; wins more in simulated games than its matchups explain |  |
| A | Sylveon | +14 | +14 | wins 22/40 1-on-1s (best: Torkoal, Flutter Mane); countered by Gyarados, Kingambit; spread | the Oct 7 opponent took it 2/2 |
| A | Dragonite | +13 | -4 | wins 21/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Flutter Mane; Tailwind, priority; wins less in simulated games than its matchups suggest | engine spends Tailwind as a flat-value move; Multiscale + Extreme Speed under Tailwind is the real plan; the Oct 7 opponent took it 2/2 |
| A | Incineroar | +13 | -12 | wins 15/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Iron Hands; Fake Out, Intimidate | engine rarely pivots, so Parting Shot / repeat Intimidates go unused; the Oct 7 opponent took it 3/3 at pick 1-2 |
| B | Tornadus | +10 | +0 | wins 3/40 1-on-1s (best: Great Tusk, Amoonguss); countered by Archaludon, Gholdengo; Tailwind, spread; wins more in simulated games than its matchups explain | Prankster Tailwind / Taunt valued by fixed constants in the engine |
| B | Urshifu-Rapid-Strike | +10 | +10 | wins 24/47 1-on-1s (best: Torkoal, Gyarados); countered by Archaludon, Flutter Mane; priority; not in the fitted model (first seen live Oct 8): matchup part only | not in the fitted model (first seen live Oct 8); Unseen Fist goes through Protect, which the draft value does not credit (its races have no Protect; the battle engine reads it) |
| B | Arcanine | +9 | +9 | wins 12/40 1-on-1s (best: Gholdengo, Flutter Mane); countered by Archaludon, Gyarados; Intimidate, spread, priority |  |
| B | Garchomp | +8 | +8 | wins 13/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Porygon2; spread |  |
| B | Great Tusk | +7 | +7 | wins 19/40 1-on-1s (best: Torkoal, Iron Hands); countered by Gholdengo, Flutter Mane; spread |  |
| B | Sneasler | +7 | +7 | wins 17/40 1-on-1s (best: Porygon2, Kingambit); countered by Archaludon, Gholdengo; Fake Out; wins more in simulated games than its matchups explain |  |
| B | Ursaluna | +6 | +6 | wins 19/40 1-on-1s (best: Torkoal, Flutter Mane); countered by Archaludon, Gholdengo; spread | our drafter's specialist pick in a loss |
| B | Basculegion | +5 | +5 | wins 18/40 1-on-1s (best: Porygon2, Milotic); countered by Archaludon, Gyarados; priority |  |
| B | Chien-Pao | +3 | +3 | wins 15/40 1-on-1s (best: Gholdengo, Flutter Mane); countered by Torkoal, Gyarados; priority |  |
| B | Wo-Chien | +1 | +1 | wins 18/40 1-on-1s (best: Gholdengo, Gyarados); countered by Archaludon, Torkoal |  |
| B | Landorus-Therian | -0 | -0 | wins 19/40 1-on-1s (best: Gholdengo, Torkoal); countered by Gyarados, Porygon2; Intimidate, spread |  |
| B | Tyranitar | -0 | -0 | wins 18/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Iron Hands; spread |  |
| B | Pelipper | -1 | -1 | wins 18/40 1-on-1s (best: Torkoal, Incineroar); countered by Archaludon, Flutter Mane; Tailwind |  |
| C | Rillaboom | -5 | -5 | wins 20/40 1-on-1s (best: Flutter Mane, Gyarados); countered by Archaludon, Torkoal; Fake Out; wins more in simulated games than its matchups explain |  |
| C | Dondozo | -6 | -6 | wins 15/40 1-on-1s (best: Torkoal, Flutter Mane); countered by Archaludon, Iron Hands; spread | no Commander (no Tatsugiri in the catalog); this set is Unaware, which the race model does credit |
| C | Farigiraf | -7 | -7 | wins 10/45 1-on-1s (best: Torkoal, Tornadus); countered by Archaludon, Gholdengo; Trick Room, spread; not in the fitted model (first seen live Oct 8): matchup part only | engine sets Trick Room whether or not its own team wants it; Armor Tail blocks Fake Out and other priority into its side |
| C | Indeedee-F | -8 | -8 | wins 12/40 1-on-1s (best: Torkoal, Iron Hands); countered by Gholdengo, Flutter Mane; redirection, spread |  |
| C | Thundurus | -8 | -8 | wins 13/40 1-on-1s (best: Gholdengo, Gyarados); countered by Archaludon, Flutter Mane |  |
| C | Salamence | -8 | -8 | wins 12/40 1-on-1s (best: Torkoal, Arcanine); countered by Archaludon, Gholdengo; Intimidate, Tailwind |  |
| C | Baxcalibur | -8 | -8 | wins 26/40 1-on-1s (best: Torkoal, Gyarados); countered by Archaludon, Gholdengo; priority; wins less in simulated games than its matchups suggest | our drafter's specialist pick in two losses |
| C | Scream Tail | -8 | -8 | wins 2/40 1-on-1s (best: Salamence, Sableye); countered by Archaludon, Gholdengo; spread |  |
| C | Talonflame | -9 | -9 | wins 8/44 1-on-1s (best: Sylveon, Tornadus); countered by Archaludon, Gholdengo; Tailwind; not in the fitted model (first seen live Oct 8): matchup part only |  |
| C | Iron Treads | -9 | -9 | wins 19/44 1-on-1s (best: Gholdengo, Flutter Mane); countered by Archaludon, Torkoal; spread; not in the fitted model (first seen live Oct 8): matchup part only |  |
| C | Iron Valiant | -10 | -10 | wins 17/44 1-on-1s (best: Iron Hands, Gyarados); countered by Archaludon, Gholdengo; not in the fitted model (first seen live Oct 8): matchup part only |  |
| C | Hatterene | -10 | -10 | wins 10/40 1-on-1s (best: Torkoal, Tornadus); countered by Archaludon, Gholdengo; Trick Room, spread; wins less in simulated games than its matchups suggest | engine sets Trick Room whether or not its own team wants it |
| C | Amoonguss | -14 | -24 | wins 9/40 1-on-1s (best: Wo-Chien, Rillaboom); countered by Archaludon, Gholdengo; redirection, sleep move | engine redirects only to save a partner from a KO; Spore + Rage Powder do more in good hands |
| C | Ting-Lu | -14 | -14 | wins 17/40 1-on-1s (best: Gholdengo, Torkoal); countered by Archaludon, Iron Hands; spread; wins less in simulated games than its matchups suggest |  |
| D | Whimsicott | -17 | -27 | wins 6/40 1-on-1s (best: Iron Hands, Great Tusk); countered by Gholdengo, Torkoal; Tailwind, screens | Prankster Tailwind / Encore valued by fixed constants in the engine |
| D | Grimmsnarl | -19 | -19 | wins 8/40 1-on-1s (best: Porygon2, Milotic); countered by Archaludon, Gholdengo; screens | screens valued by fixed constants in the engine |
| D | Lucario | -20 | -20 | wins 10/40 1-on-1s (best: Flutter Mane, Kingambit); countered by Gholdengo, Torkoal; priority; wins less in simulated games than its matchups suggest | our drafter's specialist pick in two losses |
| D | Cresselia | -22 | -22 | wins 15/40 1-on-1s (best: Archaludon, Iron Hands); countered by Gholdengo, Torkoal; spread |  |
| D | Iron Bundle | -23 | -23 | wins 16/40 1-on-1s (best: Gyarados, Milotic); countered by Archaludon, Torkoal; wins less in simulated games than its matchups suggest |  |
| D | Toxapex | -23 | -23 | wins 9/44 1-on-1s (best: Tornadus, Arcanine); countered by Archaludon, Gholdengo; not in the fitted model (first seen live Oct 8): matchup part only |  |
| D | Klefki | -37 | -37 | wins 7/40 1-on-1s (best: Dragonite, Tornadus); countered by Archaludon, Gholdengo; spread, screens; wins less in simulated games than its matchups suggest | screens valued by fixed constants in the engine |
| D | Sableye | -45 | -45 | wins 0/40 1-on-1s; countered by Archaludon, Gholdengo; Fake Out; wins less in simulated games than its matchups suggest | Prankster Will-O-Wisp / Taunt valued by fixed constants in the engine |
