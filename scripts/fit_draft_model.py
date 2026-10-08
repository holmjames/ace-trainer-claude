"""Fit the team model behind the counter-aware drafter and Team Preview from simulator games.

    python sim/draft_eval.py collect random random --games 16000 --out <dir>/train_random.jsonl
    python scripts/fit_draft_model.py <dir>/train_random.jsonl [more.jsonl ...] --write

Plain language: each simulated game gives two rosters (and the four each side brought) and a winner. For every
game we compute the team features of agent/pokemon/matchup.py (answers to each opposing card, average race
margin, best-four strength, spread pressure, speed control, doubles roles) plus one indicator per catalog card,
and fit a logistic regression: P(p1 wins) = sigmoid(w . features). Two models are fitted:

- ``draft``: six against six (the rosters), used to score finished drafts in the draft search;
- ``preview``: four against four (what each side brought) plus the margin of the opening (turns 1-2 of the two
  leads each side sent out, ``matchup.opening``), used at Team Preview.

The per-card terms ("this card wins more than its matchups explain") are kept only for cards in the catalog,
keyed by the card's fingerprint, and shrunk toward zero by a stronger penalty than the matchup weights, so the
model leans on matchups and stays sensible for a card it has never seen. Pure Python (no numpy in the agent's
environment): Newton's method with an L2 penalty, a few seconds per iteration.

Without ``--write`` it only reports held-out log-loss and accuracy against simpler models, so a change can be
judged before it is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.pokemon import data, matchup  # noqa: E402

CATALOG = ROOT / "data" / "cards.json"


def load_games(paths: list[str]) -> list[dict]:
    games = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            if game.get("winner") in ("p1", "p2"):
                games.append(game)
    return games


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (the system is small and positive definite)."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        p = a[col][col] or 1e-12
        for r in range(col + 1, n):
            f = a[r][col] / p
            if f:
                row_r, row_c = a[r], a[col]
                for k in range(col, n + 1):
                    row_r[k] -= f * row_c[k]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = a[r][n] - sum(a[r][k] * out[k] for k in range(r + 1, n))
        out[r] = s / (a[r][r] or 1e-12)
    return out


def fit_logistic(rows: list[list[tuple[int, float]]], y: list[int], dim: int, l2: list[float], iters: int = 30) -> list[float]:
    """L2-penalised logistic regression without intercept (the features are antisymmetric). ``rows`` are sparse
    (index, value) lists."""
    w = [0.0] * dim
    for _ in range(iters):
        grad = [l2[k] * w[k] for k in range(dim)]
        hess = [[0.0] * dim for _ in range(dim)]
        for k in range(dim):
            hess[k][k] = l2[k]
        for xs, yi in zip(rows, y):
            z = sum(w[k] * v for k, v in xs)
            p = sigmoid(z)
            r, s = p - yi, p * (1 - p)
            for k, v in xs:
                grad[k] += r * v
                hk, sv = hess[k], s * v
                for k2, v2 in xs:
                    hk[k2] += sv * v2
        step = solve(hess, grad)
        w = [wk - sk for wk, sk in zip(w, step)]
        if max(abs(s) for s in step) < 1e-7:
            break
    return w


def evaluate(rows, y, w) -> tuple[float, float]:
    loss = acc = 0.0
    for xs, yi in zip(rows, y):
        p = min(1 - 1e-9, max(1e-9, sigmoid(sum(w[k] * v for k, v in xs))))
        loss -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
        acc += (p > 0.5) == (yi == 1)
    return loss / len(y), acc / len(y)


class Encoder:
    """Game -> sparse feature row for one of the two models."""

    def __init__(self, cards: list[dict], *, kind: str, alpha_tw: float, alpha_tr: float, cap: float) -> None:
        self.kind = kind
        self.units = [matchup.unit(c) for c in cards]
        self.by_species = {data.to_id(u.species): i for i, u in enumerate(self.units)}
        model = {**matchup.DEFAULT_MODEL, "alpha_tw": alpha_tw, "alpha_tr": alpha_tr, "answer_cap": cap, "cards": {}}
        self.board = matchup.Board(self.units, model)
        self.n_feat = len(matchup.FEATURES) + (1 if kind == "preview" else 0)
        self.dim = self.n_feat + len(self.units)

    def names(self) -> list[str]:
        base = list(matchup.FEATURES)
        if self.kind == "preview":
            base.append("opening")  # turns 1-2 of the two leads each side opened with (matchup.opening)
        return base + [f"card:{u.species}" for u in self.units]

    def row(self, game: dict, *, use_cards: bool = True, use_matchup: bool = True) -> list[tuple[int, float]] | None:
        key = "rosters" if self.kind == "draft" else "teams"
        try:
            p1 = [self.by_species[data.to_id(s)] for s in game[key]["p1"]]
            p2 = [self.by_species[data.to_id(s)] for s in game[key]["p2"]]
        except KeyError:
            return None  # a card not in this catalog
        out: list[tuple[int, float]] = []
        if use_matchup:
            feats = self.board.features(p1, p2)
            if self.kind == "preview":  # leads are listed first
                feats = feats + [matchup.opening([self.units[i] for i in p1[:2]], [self.units[i] for i in p2[:2]]).margin]
            out += [(k, v) for k, v in enumerate(feats) if v]
        if use_cards:
            for i in p1:
                out.append((self.n_feat + i, 1.0))
            for i in p2:
                out.append((self.n_feat + i, -1.0))
        return out


def fit_kind(games: list[dict], cards: list[dict], kind: str, *, alpha_tw: float, alpha_tr: float, cap: float,
             l2_feat: float, l2_card: float, holdout: float, seed: int, report: bool = True) -> dict:
    enc = Encoder(cards, kind=kind, alpha_tw=alpha_tw, alpha_tr=alpha_tr, cap=cap)
    rng = random.Random(seed)
    order = list(range(len(games)))
    rng.shuffle(order)
    cut = int(len(order) * (1 - holdout))
    train_idx, test_idx = order[:cut], order[cut:]
    y_all = [1 if g["winner"] == "p1" else 0 for g in games]
    results = {}
    variants = {"cards only": (True, False), "matchup only": (False, True), "matchup + cards": (True, True)}
    final_w = None
    for name, (use_cards, use_matchup) in variants.items():
        rows = [enc.row(g, use_cards=use_cards, use_matchup=use_matchup) for g in games]
        keep_train = [i for i in train_idx if rows[i] is not None]
        keep_test = [i for i in test_idx if rows[i] is not None]
        l2 = [l2_feat] * enc.n_feat + [l2_card] * len(enc.units)
        t0 = time.monotonic()
        w = fit_logistic([rows[i] for i in keep_train], [y_all[i] for i in keep_train], enc.dim, l2)
        tr_loss, tr_acc = evaluate([rows[i] for i in keep_train], [y_all[i] for i in keep_train], w)
        te_loss, te_acc = evaluate([rows[i] for i in keep_test], [y_all[i] for i in keep_test], w) if keep_test else (float("nan"), float("nan"))
        results[name] = {"train_logloss": round(tr_loss, 4), "test_logloss": round(te_loss, 4), "test_accuracy": round(te_acc, 4),
                         "n_train": len(keep_train), "n_test": len(keep_test), "seconds": round(time.monotonic() - t0, 1)}
        if report:
            print(f"  [{kind}] {name:16s} train {tr_loss:.4f}  test {te_loss:.4f}  acc {100 * te_acc:.1f}%  "
                  f"({len(keep_train)}/{len(keep_test)} games, {time.monotonic() - t0:.0f} s)", flush=True)
        if use_cards and use_matchup:
            final_w = w
    return {"encoder": enc, "weights": final_w, "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--alpha-tw", type=float, default=0.25)
    parser.add_argument("--alpha-tr", type=float, default=0.25)
    parser.add_argument("--cap", type=float, default=0.3)
    parser.add_argument("--l2-feat", type=float, default=2.0)
    parser.add_argument("--l2-card", type=float, default=40.0)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kinds", default="draft,preview")
    parser.add_argument("--write", action="store_true", help="refit on ALL games and write data/draft_model.json")
    args = parser.parse_args(argv)
    cards = json.loads(CATALOG.read_text(encoding="utf-8"))
    games = load_games(args.files)
    print(f"{len(games)} decided games from {len(args.files)} file(s); catalog {len(cards)} cards")
    blob: dict = {"note": "Fitted by scripts/fit_draft_model.py on simulator games (code brain on both sides). "
                          "Weights are log-odds per unit of each team feature in agent/pokemon/matchup.py; "
                          "'cards' are per-card residual strengths keyed by card fingerprint."}
    for kind in args.kinds.split(","):
        fitted = fit_kind(games, cards, kind, alpha_tw=args.alpha_tw, alpha_tr=args.alpha_tr, cap=args.cap,
                          l2_feat=args.l2_feat, l2_card=args.l2_card, holdout=args.holdout, seed=args.seed)
        enc, w = fitted["encoder"], fitted["weights"]
        if args.write:
            final = fit_kind(games, cards, kind, alpha_tw=args.alpha_tw, alpha_tr=args.alpha_tr, cap=args.cap,
                             l2_feat=args.l2_feat, l2_card=args.l2_card, holdout=0.0, seed=args.seed, report=False)
            w = final["weights"]
        names = enc.names()
        weights = {n: round(v, 4) for n, v in zip(names[:enc.n_feat], w[:enc.n_feat])}
        print(f"  [{kind}] weights: " + ", ".join(f"{k}={v:+.3f}" for k, v in weights.items()))
        residuals = sorted(((w[enc.n_feat + i], u) for i, u in enumerate(enc.units)), key=lambda t: -t[0])
        print(f"  [{kind}] card residuals: " + ", ".join(f"{u.species} {v:+.2f}" for v, u in residuals))
        section = {"alpha_tw": args.alpha_tw, "alpha_tr": args.alpha_tr, "answer_cap": args.cap,
                   "weights": {k: v for k, v in weights.items() if k != "opening"},
                   "cards": {u.key: {"species": u.species, "s": round(v, 4)} for v, u in residuals},
                   "fit": {**fitted["results"], "games": len(games), "l2_feat": args.l2_feat, "l2_card": args.l2_card}}
        if kind == "preview":
            section["opening_weight"] = weights.get("opening", 0.0)
        blob[kind] = section
    if args.write:
        out = ROOT / "data" / "draft_model.json"
        out.write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
