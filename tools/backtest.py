"""tools/backtest.py -- measure the model with the competition's own metric.

The server's answer key for the practice cases is not in the kit, so this tool measures
the model four ways and prints all four:

  real       Ground truth that does exist: each case's later reads.  For every pair of
             reads (j, k) with k after j, build the sub-case the house would have cut at
             read j (history up to j, checkpoint = read j's time, target = read k's time),
             forecast it, and score against read k with the competition's pinball loss --
             the same formula the server uses, on the logit scale, over every trophy and
             level.  On the practice histories this is degenerate: every read before the
             first refresh is the same page, so the sub-cases are flat and persistence is
             optimal by construction.  On a final-shaped case (refreshes in the history)
             this is the real test.

  practice   A reconstruction of the practice answers from the aggregates the kit
             documents: the page went from 638 players tracked and 943 trophies earned to
             3,868 tracked and 4,276 earned, and every rate but the one at 100% fell.  The
             one 100% trophy accounts for the whole tracked base, so the other 54 went
             from 943-638=305 earned to 4,276-3,868=408, i.e. their rates fell about 4.5
             times.  How those 408 split among the 54 is not published, so two splits
             bracket it: `flat` (each count grows by the same factor) and `depth` (the
             counts grow more where a fresh player reaches first).  Neither is the answer
             key; together they say how a forecast that expects the refresh does, and how
             much of the loss is unavoidable spread in the truth.

  synthetic  A parametric launch built to the documented dynamics: tracked players arrive
             over a day (638 at 04:30Z, 3,868 at 05:00Z), each has played since arriving,
             the chance of having earned a trophy rises with play time at a tier-dependent
             timescale, Exophase republishes every few hours (2-decimal printing, so reads
             inside a plateau are identical), and the truth is the page at the target.
             Cases are cut like the practice (90 minutes) and like the final (6,510
             minutes).  The world is an assumption, not data; it is the only place a
             week-long horizon can be measured at all, so its numbers are reported as
             such.

  sweep      A grid over the sigma knobs, scored on the practice reconstruction and the
             synthetic set at both horizons.  This is how the numbers in model.PARAMS were
             chosen.

Usage:
    python3 tools/backtest.py real     --input <dir> [--pairs 6]
    python3 tools/backtest.py practice --input <dir>
    python3 tools/backtest.py synthetic [--worlds 12] [--seed 7]
    python3 tools/backtest.py sweep    [--worlds 8]
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model  # noqa: E402

LEVELS = model.LEVELS
Z = model.Z
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# The documented launch-night page, before and after Exophase's first refresh.
TRACKED_BEFORE = 638.0
EARNED_BEFORE = 943.0
TRACKED_AFTER = 3868.0
EARNED_AFTER = 4276.0


# -------------------------------------------------------------------------- scoring

def pinball(q, truth):
    """2*max(tau*d, (tau-1)*d) per level, d = logit(truth) - logit(q); mean over levels."""
    zy = model.logit(truth)
    total = 0.0
    for tau, qv in zip(LEVELS, q):
        d = zy - model.logit(qv)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def score(rows, truths, ids):
    """Mean pinball loss, typical miss in points, 90% coverage, per-level coverage."""
    by_id = {row["id"]: row["q"] for row in rows}
    losses, misses = [], []
    hits = [0] * len(LEVELS)
    covered = 0
    for tid, truth in zip(ids, truths):
        if truth is None or tid not in by_id:
            continue
        q = by_id[tid]
        y = model.clamp_pct(truth)
        losses.append(pinball(q, truth))
        misses.append(abs(y - q[2]))
        covered += 1 if (q[0] <= y <= q[4]) else 0
        for k, qv in enumerate(q):
            hits[k] += 1 if qv <= y else 0
    n = len(losses)
    if not n:
        return {"loss": 0.0, "miss": 0.0, "coverage": 0.0, "levels": [0.0] * len(LEVELS)}
    return {"loss": sum(losses) / n, "miss": sum(misses) / n, "coverage": covered / n,
            "levels": [h / n for h in hits]}


def baseline_rows(case):
    """The house baseline, number for number: persistence + random-walk spread."""
    hist = case.get("history") or []
    rates = (hist[-1].get("rates") if hist else None) or []
    days = max(float(case.get("horizon_minutes") or 0.0), 0.0) / 1440.0
    spread = 0.5 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = rates[i] if i < len(rates) else None
        lg = model.logit(rate if rate is not None else 0.01)
        rows.append({"id": trophy.get("id", i + 1),
                     "q": [model.half_up(model.inv_logit(lg + z * spread)) for z in Z]})
    return rows


def model_rows(case, params=None):
    return model.forecast_case(case, params)["trophies"]


def fmt(tag, res):
    levels = " ".join("%.2f" % v for v in res["levels"])
    return ("  %-22s loss %.4f  miss %6.3f pp  cov %.3f  levels %s"
            % (tag, res["loss"], res["miss"], res["coverage"], levels))


# --------------------------------------------------------------------- case loading

def load_cases(directory):
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(Path(directory).glob("*.in"))]


def iso(hours):
    return (LAUNCH + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def truncate(case, j, k):
    """The sub-case cut at read j, its target read k's time."""
    hist = case["history"]
    if k <= j:
        return None
    sub = dict(case)
    sub["history"] = hist[:j + 1]
    sub["at"] = hist[j].get("at") or hist[j].get("slot")
    sub["target"] = hist[k].get("at") or hist[k].get("slot")
    sub["horizon_minutes"] = (model.parse_ts(sub["target"])
                              - model.parse_ts(sub["at"])) / 60.0
    sub["id"] = "%s@%d>%d" % (case.get("id", "?"), j, k)
    return sub


# --------------------------------------------------------------------------- 1. real

def backtest_real(cases, pairs=6, params=None, verbose=True):
    out = []
    for case in cases:
        hist = case.get("history") or []
        ids = [t["id"] for t in case.get("trophies") or []]
        n = len(hist)
        if n < 2:
            continue
        cuts = [(j, k) for k in range(max(1, n - pairs), n) for j in range(k)]
        got = []
        for j, k in cuts:
            sub = truncate(case, j, k)
            truths = hist[k].get("rates")
            if sub is None or not any(t is not None for t in truths or []):
                continue
            got.append((score(model_rows(sub, params), truths, ids),
                        score(baseline_rows(sub), truths, ids)))
        if not got:
            continue
        mean = lambda key, which: sum(g[which][key] for g in got) / len(got)
        row = {"case": case.get("id"), "pairs": len(got),
               "model_loss": mean("loss", 0), "baseline_loss": mean("loss", 1),
               "model_miss": mean("miss", 0), "baseline_miss": mean("miss", 1),
               "model_coverage": mean("coverage", 0),
               "baseline_coverage": mean("coverage", 1)}
        out.append(row)
        if verbose:
            print("  %-6s pairs=%3d  model %.4f  baseline %.4f  miss %.3f pp  cov %.3f"
                  % (row["case"], row["pairs"], row["model_loss"], row["baseline_loss"],
                     row["model_miss"], row["model_coverage"]))
    return out


# ----------------------------------------------------------------------- 2. practice

def practice_truths(case, kind="flat"):
    """Reconstruct the answering page (3,868 tracked, 4,276 earned) per trophy."""
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    before = [max(0, int(round((r or 0.0) * TRACKED_BEFORE / 100.0))) for r in rates]
    full = [i for i, r in enumerate(rates) if (r or 0.0) >= model.PARAMS["full_pct"]]
    held_before = sum(before[i] for i in full)
    others_before = sum(before) - held_before
    held_after = TRACKED_AFTER * len(full)          # the 100% trophy stays 100%
    others_after = max(EARNED_AFTER - held_after, 1.0)
    factor = others_after / max(others_before, 1.0)
    others = [i for i in range(len(before)) if i not in full]
    if kind == "flat":
        mult = {i: 1.0 for i in others}
    else:
        # a fresh player reaches the early (common) trophies first: weight by rate,
        # scaled so the tilt keeps the documented total of the other trophies
        weights = {i: max(rates[i] or 0.01, 0.01) ** 0.5 for i in others}
        mean_w = sum(weights.values()) / max(len(others), 1)
        mult = {i: weights[i] / mean_w for i in others}
    counts = []
    for i, count in enumerate(before):
        if i in full:
            counts.append(TRACKED_AFTER)
        else:
            counts.append(count * factor * mult[i])
    truths = [round(100.0 * c / TRACKED_AFTER, 2) for c in counts]
    return [t["id"] for t in case["trophies"]], truths


def backtest_practice(cases, params=None, verbose=True):
    out = []
    for case in cases:
        fc = model.Forecaster(case, params)
        rows = fc.forecast()
        for kind in ("flat", "depth"):
            ids, truths = practice_truths(case, kind)
            got = score(rows, truths, ids)
            base = score(baseline_rows(case), truths, ids)
            out.append({"case": case.get("id"), "truth": kind, "family": fc.family,
                        "loss": got["loss"], "baseline": base["loss"],
                        "miss": got["miss"], "coverage": got["coverage"],
                        "levels": got["levels"]})
            if verbose:
                print("  %-4s %-6s family=%-12s model %.4f  baseline %.4f  miss %.3f pp"
                      "  cov %.3f  levels %s"
                      % (case.get("id"), kind, fc.family, got["loss"], base["loss"],
                         got["miss"], got["coverage"],
                         " ".join("%.2f" % v for v in got["levels"])))
    return out


# ---------------------------------------------------------------------- 3. synthetic

TIER_TAU = {"bronze": (8.0, 55.0), "silver": (25.0, 90.0), "gold": (40.0, 140.0),
            "platinum": (80.0, 280.0)}


class World:
    """A parametric launch that starts from the documented launch night.

    The tracked base is not a smooth curve: the house saw 638 players tracked from just
    before launch through its 04:00Z read, and 3,868 on its next read -- Exophase's crawler
    discovers the player base in steps, hours apart.  The world reproduces that step at a
    jittered hour, then lets the base grow towards a ceiling over the following days.  Each
    tracked player has played since they arrived (the first cohort a little longer), and
    the chance of having earned a trophy rises with play time at a tier-dependent
    timescale, so a bigger base of fresh players dilutes every rate while the old cohort
    keeps earning.  Every world is calibrated so its page on launch night shows the real
    case's rates, jittered per world.
    """

    def __init__(self, seed, trophies, seed_rates=None):
        rng = random.Random(seed)
        self.rng = rng
        self.trophies = trophies
        self.n = len(trophies)
        self.step_h = 4.67 + rng.uniform(-0.25, 0.25)   # the crawler's first step
        self.ceiling = math.exp(rng.uniform(math.log(4.0e4), math.log(4.0e5)))
        self.tau_base = rng.uniform(30.0, 90.0)
        self.speed = math.exp(rng.uniform(math.log(0.7), math.log(1.6)))
        self.age0 = rng.uniform(1.0, 3.5)     # playtime of the first tracked players
        self.refresh = rng.choice((2.0, 3.0, 4.0, 6.0))
        self.tau = []
        self.share = [1.0] * self.n
        rates = list(seed_rates or [None] * self.n)
        for i, trophy in enumerate(trophies):
            lo, hi = TIER_TAU.get(trophy.get("tier"), TIER_TAU["bronze"])
            self.tau.append(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        first = self._rates(TRACKED_BEFORE, -0.167)
        for i, trophy in enumerate(trophies):
            rate = rates[i]
            if rate is None:
                rate = 1.0
            if (rate or 0.0) >= model.PARAMS["full_pct"] or trophy.get("id") == 2:
                self.share[i] = 0.999
                self.tau[i] = 0.25
                continue
            want = max(rate * rng.uniform(0.7, 1.4), 0.04) / 100.0
            reach = first[i] / 100.0
            self.share[i] = min(0.995, want / max(reach, 1e-9))

    def tracked(self, hours):
        if hours < self.step_h:
            return TRACKED_BEFORE
        return TRACKED_AFTER + (self.ceiling - TRACKED_AFTER) * (
            1.0 - math.exp(-(hours - self.step_h) / self.tau_base))

    def _arrivals(self, hours, steps=160):
        out = [(-self.age0, TRACKED_BEFORE)]
        if hours >= self.step_h:
            out.append((self.step_h, TRACKED_AFTER - TRACKED_BEFORE))
            prev = TRACKED_AFTER
            for k in range(1, steps + 1):
                h = self.step_h + (hours - self.step_h) * k / steps
                now = self.tracked(h)
                out.append((h, now - prev))
                prev = now
        return out

    def _rates(self, total, hours):
        arrivals = self._arrivals(max(hours, 0.0) + 1e-9)
        out = []
        for i in range(self.n):
            earned = 0.0
            for arrived, count in arrivals:
                if count <= 0.0:
                    continue
                # the first cohort arrived `age0` hours before launch, so `hours - arrived`
                # is each player's play time; the fresh arrivals start at zero
                played = max(hours - arrived, 0.0) * self.speed
                earned += count * self.share[i] * (1.0 - math.exp(-played / self.tau[i]))
            out.append(100.0 * earned / max(total, 1.0))
        return out

    def page(self, hours):
        """The rates Exophase shows at `hours`, i.e. its last refresh, printed to 2 dp.

        The first refresh is the crawler's step: every read before it is the launch
        snapshot, unchanged -- exactly the plateau the house recorded on launch night.
        """
        if hours < self.step_h:
            stamp = 0.0
        else:
            stamp = self.step_h + math.floor((hours - self.step_h) / self.refresh) * self.refresh
        memo = self.__dict__.setdefault("_memo", {})
        key = round(stamp, 6)
        if key not in memo:
            memo[key] = [round(r, 2) for r in self._rates(self.tracked(stamp), stamp)]
        return list(memo[key])

    def case(self, checkpoint_h, horizon_h, cadence_min, case_id="syn"):
        hist = []
        step = cadence_min / 60.0
        h = -0.1667                      # the house's first read, just before launch
        while h < checkpoint_h - 1e-9:
            total = self.tracked(h)
            rates = self.page(h)
            hist.append({"slot": iso(h), "at": iso(h),
                         "players_tracked": int(round(total)),
                         "trophies_earned": int(round(sum(rates) * total / 100.0)),
                         "rates": rates})
            h += step
        return {"schema": "makefun/gta6-trophies-case/v1", "id": case_id,
                "at": iso(checkpoint_h), "target": iso(checkpoint_h + horizon_h),
                "horizon_minutes": horizon_h * 60.0, "levels": list(LEVELS),
                "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                         "release_date": "2026-11-19",
                         "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
                "trophies": self.trophies, "history": hist, "notes": "synthetic"}


SHAPES = (("90 min", "short", 1.5, 30.0), ("4.5 days", "long", 108.5, 10.0))


def backtest_synthetic(trophies, worlds=12, seed=7, params=None, verbose=True,
                       seed_rates=None, shapes=SHAPES):
    totals = {}
    for label, _kind, _hz, _cad in shapes:
        totals[label] = {"model": [], "base": [], "miss": [], "cov": [], "levels": []}
    ids = [t["id"] for t in trophies]
    for w in range(worlds):
        world = World(seed + w, trophies, seed_rates)
        for label, kind, hz, cad in shapes:
            # short: cut on launch night, the target just after the crawler's step;
            # long: cut two and a half days in, the target a week after launch
            if kind == "short":
                checkpoint = world.step_h - 1.45 + (w % 4) * 0.05
            else:
                checkpoint = 60.0 + (w % 5) * 1.7
            case = world.case(checkpoint, hz, cad)
            truth = world.page(checkpoint + hz)
            got = score(model_rows(case, params), truth, ids)
            base = score(baseline_rows(case), truth, ids)
            slot = totals[label]
            slot["model"].append(got["loss"])
            slot["base"].append(base["loss"])
            slot["miss"].append(got["miss"])
            slot["cov"].append(got["coverage"])
            slot["levels"].append(got["levels"])
    out = {}
    for label, slot in totals.items():
        if not slot["model"]:
            continue
        n = len(slot["model"])
        out[label] = {"model_loss": sum(slot["model"]) / n,
                      "baseline_loss": sum(slot["base"]) / n,
                      "miss": sum(slot["miss"]) / n,
                      "coverage": sum(slot["cov"]) / n,
                      "levels": [sum(v[k] for v in slot["levels"]) / n
                                 for k in range(len(LEVELS))]}
        if verbose:
            print("  %-8s model %.4f  baseline %.4f  miss %6.3f pp  cov %.3f  levels %s"
                  % (label, out[label]["model_loss"], out[label]["baseline_loss"],
                     out[label]["miss"], out[label]["coverage"],
                     " ".join("%.2f" % v for v in out[label]["levels"])))
    return out


# -------------------------------------------------------------------------- 4. sweep

def sweep(cases, trophies, worlds=8, seed=7, seed_rates=None):
    """Two grids: the sigma knobs on the synthetic set, the refresh prior on practice."""
    print("  -- sigma (synthetic worlds, both horizons) --")
    rows = []
    for infl in (0.80, 1.00, 1.30):
        for rw in (0.30, 0.42, 0.55, 0.70):
            params = {"infl": infl, "rw_per_day": rw}
            syn = backtest_synthetic(trophies, worlds=worlds, seed=seed, params=params,
                                     verbose=False, seed_rates=seed_rates)
            short = syn.get("90 min", {})
            long_ = syn.get("4.5 days", {})
            total = 0.5 * (short.get("model_loss", 0.0) + long_.get("model_loss", 0.0))
            rows.append({"params": params, "short": short, "long": long_,
                         "total": total})
    rows.sort(key=lambda r: r["total"])
    for row in rows:
        p = row["params"]
        print("  syn %.4f  90m %.4f (cov %.3f)  4.5d %.4f (cov %.3f)   infl=%.2f rw=%.2f"
              % (row["total"], row["short"].get("model_loss", 0.0),
                 row["short"].get("coverage", 0.0), row["long"].get("model_loss", 0.0),
                 row["long"].get("coverage", 0.0), p["infl"], p["rw_per_day"]))

    print("  -- launch prior (practice reconstruction) --")
    for prior_sd in (0.30, 0.40, 0.50, 0.60, 0.75):
        for shift in (1.20, 1.51, 1.80):
            prac = backtest_practice(cases, params={"prior_sd": prior_sd,
                                                    "prior_shift": shift}, verbose=False)
            flat = [p["loss"] for p in prac if p["truth"] == "flat"]
            depth = [p["loss"] for p in prac if p["truth"] == "depth"]
            print("  prior_sd=%.2f shift=%.2f   flat %.4f  depth %.4f  mean %.4f"
                  % (prior_sd, shift, sum(flat) / len(flat), sum(depth) / len(depth),
                     (sum(flat) + sum(depth)) / (len(flat) + len(depth))))
    return rows


# --------------------------------------------------------------------------- helpers

def default_input():
    for path in (os.environ.get("MAKEFUN_KIT_INPUT"),
                 "/root/.hermes/nft-hunter/makefun/gta6-trophy-rarity-forecast-01-kit/input",
                 str(ROOT / "input")):
        if path and Path(path).is_dir() and any(Path(path).glob("*.in")):
            return str(path)
    return str(ROOT / "input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("real", "practice", "synthetic", "sweep"))
    ap.add_argument("--input", default=None)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    directory = args.input or default_input()
    cases = load_cases(directory) if Path(directory).is_dir() else []
    print("%s: %d case(s) from %s" % (args.mode, len(cases), directory))

    if args.mode == "real":
        backtest_real(cases, pairs=args.pairs)
        return
    if args.mode == "practice":
        backtest_practice(cases)
        return

    trophies = cases[0]["trophies"] if cases else []
    seed_rates = ((cases[0].get("history") or [{}])[-1].get("rates") if cases else None)
    if args.mode == "synthetic":
        backtest_synthetic(trophies, worlds=args.worlds, seed=args.seed,
                           seed_rates=seed_rates)
    else:
        sweep(cases, trophies, worlds=args.worlds, seed=args.seed, seed_rates=seed_rates)


if __name__ == "__main__":
    main()
