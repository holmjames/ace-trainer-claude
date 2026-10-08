# Card tiers (live catalog)

Built by `scripts/build_tiers.py` from the fitted team model; the drafter's base value per card. Rating = average worth on a team in 1/100 log-odds of winning (sim rating + a small prior where the code-only simulator is known to misplay the card). Rebuild after refitting `data/draft_model.json`.

| Tier | Card | Rating | Sim | Why | Flag / live |
|---|---|---:|---:|---|---|
| S | Archaludon | +61 | +61 | wins 27/40 1-on-1s (e.g. Rillaboom, Grimmsnarl), loses 4; wins more in simulated games than its matchups explain | the search bot took it 3/4 |
| S | Gholdengo | +53 | +53 | wins 18/40 1-on-1s (e.g. Hatterene, Sneasler), loses 15; spread; wins more in simulated games than its matchups explain | the search bot took it 2/2 |
| A | Torkoal | +25 | +25 | wins 16/40 1-on-1s (e.g. Scream Tail, Klefki), loses 20; spread, sleep move; wins more in simulated games than its matchups explain |  |
| A | Flutter Mane | +23 | +23 | wins 20/40 1-on-1s (e.g. Dragonite, Gholdengo), loses 11; spread; wins more in simulated games than its matchups explain |  |
| A | Iron Hands | +21 | +21 | wins 23/40 1-on-1s (e.g. Ursaluna, Kingambit), loses 7; Fake Out | the search bot took it 2/3 |
| A | Gyarados | +18 | +18 | wins 26/40 1-on-1s (e.g. Grimmsnarl, Scream Tail), loses 8; Intimidate, spread |  |
| A | Porygon2 | +16 | +16 | wins 16/40 1-on-1s (e.g. Amoonguss, Iron Bundle), loses 10; Trick Room | engine sets Trick Room whether or not its own team wants it |
| A | Kingambit | +16 | +16 | wins 22/40 1-on-1s (e.g. Basculegion, Flutter Mane), loses 10; priority |  |
| A | Milotic | +14 | +14 | wins 13/40 1-on-1s (e.g. Torkoal, Salamence), loses 16; wins more in simulated games than its matchups explain |  |
| A | Sylveon | +14 | +14 | wins 22/40 1-on-1s (e.g. Sableye, Salamence), loses 8; spread | the search bot took it 2/2 |
| A | Dragonite | +13 | -4 | wins 21/40 1-on-1s (e.g. Tornadus, Torkoal), loses 7; Tailwind, priority; wins less in simulated games than its matchups suggest | engine spends Tailwind as a flat-value move; Multiscale + Extreme Speed under Tailwind is the real plan; the search bot took it 2/2 |
| A | Incineroar | +13 | -12 | wins 15/40 1-on-1s (e.g. Indeedee-F, Klefki), loses 16; Fake Out, Intimidate | engine rarely pivots, so Parting Shot / repeat Intimidates go unused; the search bot took it 3/3 at pick 1-2 |
| B | Tornadus | +10 | +0 | wins 3/40 1-on-1s (e.g. Great Tusk, Amoonguss), loses 30; Tailwind, spread; wins more in simulated games than its matchups explain | Prankster Tailwind / Taunt valued by fixed constants in the engine |
| B | Arcanine | +9 | +9 | wins 12/40 1-on-1s (e.g. Klefki, Whimsicott), loses 16; Intimidate, spread, priority |  |
| B | Garchomp | +8 | +8 | wins 13/40 1-on-1s (e.g. Lucario, Torkoal), loses 13; spread |  |
| B | Great Tusk | +7 | +7 | wins 19/40 1-on-1s (e.g. Grimmsnarl, Iron Hands), loses 13; spread |  |
| B | Sneasler | +7 | +7 | wins 17/40 1-on-1s (e.g. Grimmsnarl, Sylveon), loses 15; Fake Out; wins more in simulated games than its matchups explain |  |
| B | Ursaluna | +6 | +6 | wins 19/40 1-on-1s (e.g. Torkoal, Klefki), loses 14; spread | our drafter's specialist pick in a loss |
| B | Basculegion | +5 | +5 | wins 18/40 1-on-1s (e.g. Sableye, Hatterene), loses 12; priority |  |
| B | Chien-Pao | +3 | +3 | wins 15/40 1-on-1s (e.g. Basculegion, Flutter Mane), loses 6; priority |  |
| B | Wo-Chien | +1 | +1 | wins 18/40 1-on-1s (e.g. Dondozo, Milotic), loses 15 |  |
| B | Landorus-Therian | -0 | -0 | wins 19/40 1-on-1s (e.g. Klefki, Sneasler), loses 12; Intimidate, spread |  |
| B | Tyranitar | -0 | -0 | wins 18/40 1-on-1s (e.g. Tornadus, Sableye), loses 14; spread |  |
| B | Pelipper | -1 | -1 | wins 18/40 1-on-1s (e.g. Ursaluna, Sableye), loses 13; Tailwind |  |
| C | Rillaboom | -5 | -5 | wins 20/40 1-on-1s (e.g. Sableye, Milotic), loses 10; Fake Out; wins more in simulated games than its matchups explain |  |
| C | Dondozo | -6 | -6 | wins 15/40 1-on-1s (e.g. Klefki, Arcanine), loses 14; spread | no Commander (no Tatsugiri in the catalog); this set is Unaware, which the race model does credit |
| C | Indeedee-F | -8 | -8 | wins 12/40 1-on-1s (e.g. Amoonguss, Torkoal), loses 15; redirection, spread |  |
| C | Thundurus | -8 | -8 | wins 13/40 1-on-1s (e.g. Dondozo, Gyarados), loses 17 |  |
| C | Salamence | -8 | -8 | wins 12/40 1-on-1s (e.g. Dondozo, Rillaboom), loses 19; Intimidate, Tailwind |  |
| C | Baxcalibur | -8 | -8 | wins 26/40 1-on-1s (e.g. Amoonguss, Ursaluna), loses 7; priority; wins less in simulated games than its matchups suggest | our drafter's specialist pick in two losses |
| C | Scream Tail | -8 | -8 | wins 2/40 1-on-1s (e.g. Sableye, Salamence), loses 27; spread |  |
| C | Hatterene | -10 | -10 | wins 10/40 1-on-1s (e.g. Klefki, Iron Bundle), loses 18; Trick Room, spread; wins less in simulated games than its matchups suggest | engine sets Trick Room whether or not its own team wants it |
| C | Amoonguss | -14 | -24 | wins 9/40 1-on-1s (e.g. Scream Tail, Whimsicott), loses 23; redirection, sleep move | engine redirects only to save a partner from a KO; Spore + Rage Powder do more in good hands |
| C | Ting-Lu | -14 | -14 | wins 17/40 1-on-1s (e.g. Thundurus, Sableye), loses 13; spread; wins less in simulated games than its matchups suggest |  |
| D | Whimsicott | -17 | -27 | wins 6/40 1-on-1s (e.g. Sableye, Iron Hands), loses 22; Tailwind, screens | Prankster Tailwind / Encore valued by fixed constants in the engine |
| D | Grimmsnarl | -19 | -19 | wins 8/40 1-on-1s (e.g. Sableye, Wo-Chien), loses 21; screens | screens valued by fixed constants in the engine |
| D | Lucario | -20 | -20 | wins 10/40 1-on-1s (e.g. Flutter Mane, Incineroar), loses 19; priority; wins less in simulated games than its matchups suggest | our drafter's specialist pick in two losses |
| D | Cresselia | -22 | -22 | wins 15/40 1-on-1s (e.g. Sableye, Lucario), loses 17; spread |  |
| D | Iron Bundle | -23 | -23 | wins 16/40 1-on-1s (e.g. Great Tusk, Gyarados), loses 17; wins less in simulated games than its matchups suggest |  |
| D | Klefki | -37 | -37 | wins 7/40 1-on-1s (e.g. Sableye, Tornadus), loses 28; spread, screens; wins less in simulated games than its matchups suggest | screens valued by fixed constants in the engine |
| D | Sableye | -45 | -45 | wins 0/40 1-on-1s (e.g. -), loses 34; Fake Out; wins less in simulated games than its matchups suggest | Prankster Will-O-Wisp / Taunt valued by fixed constants in the engine |
