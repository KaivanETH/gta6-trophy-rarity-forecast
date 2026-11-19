"""tools/backtest.py -- score model.py the way the server scores an entry.

Three honest evaluations, all on the competition's own metric: mean pinball loss over
every trophy and level on the logit scale, plus the typical miss in percentage points
and the 90% coverage the board reports.

  truncations  Ground truth that exists in the kit: every later read of a practice case
               is the answer to a sub-case cut at an earlier read.  On the practice
               histories nothing moves before the first republish, so these folds say
               whether a design leaves a flat stretch alone.  The house persistence
               baseline is scored on the same folds.

  practice     The practice answers are the refreshed page, and the kit documents its
               aggregates (638 -> 3,868 tracked, 943 -> 4,276 earned, every rate but the
               100% one lower).  How the 408 non-100% earned counts split over the 54
               trophies is not published, so two splits bracket it: `flat` (each count
               grows by the same factor, the maximum-entropy split) and `lr` (the same
               factors rounded to whole counts by largest remainder).  Both keep the
               documented totals exactly.  The truth is a reconstruction, not the answer
               key; it is the only short-horizon test of the republish transform there is.

  synthetic    Post-launch worlds: players arrive in steps, each trophy's rate follows a
               saturating logistic in log time, Exophase republishes every few hours so
               reads inside a plateau are identical, and two-decimal printing is applied.
               Cases are cut at 59.5 h with a 4.5-day target, the final's shape.  The
               world is an assumption, but it is the only place a week-long horizon can
               be measured at all, so its numbers are reported as such.

Every evaluation scores the candidate and the house persistence baseline (last read,
plus a random walk of 0.5*sqrt(days) on the logit) on the same folds.

Usage:
    python3 tools/backtest.py                 # all three
    python3 tools/backtest.py truncations --input <kit input dir>
    python3 tools/backtest.py practice    --input <kit input dir>
    python3 tools/backtest.py synthetic   [--worlds 12] [--seed 7]
    python3 tools/backtest.py sweep       [--worlds 8]
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model  # noqa: E402

LEVELS = model.LEVELS
Z = model.Z
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)
KIT_INPUT = Path('/root/.hermes/nft-hunter/makefun/'
                 'gta6-trophy-rarity-forecast-01-kit/input')

TRACKED_BEFORE = 638.0
EARNED_BEFORE = 943.0
TRACKED_AFTER = 3868.0
EARNED_AFTER = 4276.0


# ------------------------------------------------------------------------- scoring

def pinball(q, truth):
    """The competition's loss for one trophy: mean over the five levels."""
    zy = model.logit(truth)
    total = 0.0
    for tau, qv in zip(LEVELS, q):
        d = zy - model.logit(qv)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def score(rows, truths, ids):
    """loss, typical miss (pp), 90% coverage and per-level hit rates."""
    by_id = {row['id']: row['q'] for row in rows}
    losses, misses = [], []
    covered = 0
    hits = [0] * len(LEVELS)
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
        return {'loss': 0.0, 'miss': 0.0, 'coverage': 0.0, 'levels': [0.0] * len(LEVELS)}
    return {'loss': sum(losses) / n,
            'miss': sum(misses) / n,
            'coverage': covered / n,
            'levels': [h / n for h in hits]}


def baseline_rows(case):
    """The house persistence baseline, number for number."""
    hist = case.get('history') or []
    rates = (hist[-1].get('rates') if hist else None) or []
    days = max(float(case.get('horizon_minutes') or 0.0), 0.0) / 1440.0
    spread = 0.5 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case.get('trophies') or []):
        rate = rates[i] if i < len(rates) else None
        lg = model.logit(rate if rate is not None else 0.01)
        rows.append({'id': trophy.get('id', i + 1),
                     'q': [model.half_up(model.inv_logit(lg + z * spread)) for z in Z]})
    return rows


def model_rows(case, params=None):
    return model.forecast_case(case, params)['trophies']


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def avg(values):
    return sum(values) / len(values) if values else 0.0


def line(tag, cand, base):
    return ('  %-18s model %.4f  base %.4f  miss %6.3f pp  cov %.3f  levels %s'
            % (tag, cand['loss'], base['loss'], cand['miss'], cand['coverage'],
               ' '.join('%.2f' % v for v in cand['levels'])))


def load_cases(directory):
    directory = Path(directory) if directory else KIT_INPUT
    paths = sorted(directory.glob('*.in'))
    if not paths:
        paths = sorted(KIT_INPUT.glob('*.in'))
    return [json.loads(p.read_text(encoding='utf-8')) for p in paths]


def iso(hours):
    return (RELEASE + timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%SZ')


# --------------------------------------------------------------------- 1. truncations

def truncate(case, j, k):
    sub = dict(case)
    sub['history'] = case['history'][:j + 1]
    sub['at'] = case['history'][j].get('at') or case['history'][j].get('slot')
    sub['target'] = case['history'][k].get('at') or case['history'][k].get('slot')
    sub['horizon_minutes'] = (model.parse_ts(sub['target'])
                              - model.parse_ts(sub['at'])) / 60.0
    sub['id'] = '%s@%d>%d' % (case.get('id'), j, k)
    return sub


def backtest_truncations(cases, params=None, verbose=True):
    out = []
    for case in cases:
        hist = case.get('history') or []
        ids = [t['id'] for t in case.get('trophies') or []]
        cand, base = [], []
        for k in range(1, len(hist)):
            for j in range(k):
                sub = truncate(case, j, k)
                truths = hist[k].get('rates')
                if not any(v is not None for v in truths or []):
                    continue
                cand.append(score(model_rows(sub, params), truths, ids))
                base.append(score(baseline_rows(sub), truths, ids))
        if not cand:
            continue
        row = {'model': mean(cand, 'loss'), 'base': mean(base, 'loss'),
               'miss': mean(cand, 'miss'), 'coverage': mean(cand, 'coverage'),
               'levels': [avg([c['levels'][k] for c in cand]) for k in range(5)],
               'folds': len(cand), 'case': case.get('id')}
        out.append(row)
        if verbose:
            print('  %-4s folds=%3d  model %.4f  base %.4f  miss %6.3f pp  cov %.3f'
                  % (row['case'], row['folds'], row['model'], row['base'],
                     row['miss'], row['coverage']))
    return out


# ------------------------------------------------------------------------ 2. practice

def practice_truth(case, kind='flat'):
    """Reconstruct the answering page (3,868 tracked, 4,276 earned) per trophy."""
    rates = (case.get('history') or [{}])[-1].get('rates') or []
    counts = [max(0, int(round((r or 0.0) * TRACKED_BEFORE / 100.0))) for r in rates]
    full = [i for i, r in enumerate(rates) if (r or 0.0) >= model.PARAMS['full_pct']]
    held_before = sum(counts[i] for i in full)
    others_before = sum(counts) - held_before
    others_after = EARNED_AFTER - TRACKED_AFTER * len(full)
    if others_before <= 0.0:
        return [t['id'] for t in case['trophies']], [r for r in rates]
    factor = others_after / others_before
    others = [i for i in range(len(counts)) if i not in full]
    alloc = {}
    if kind == 'flat':
        for i in others:
            alloc[i] = counts[i] * factor
    else:
        # largest-remainder rounding of the same factors, to whole earned counts
        floors = {i: math.floor(counts[i] * factor) for i in others}
        need = int(round(others_after)) - sum(floors.values())
        order = sorted(others, key=lambda i: -(counts[i] * factor - floors[i]))
        for i in others:
            alloc[i] = floors[i] + (1 if i in set(order[:max(0, need)]) else 0)
    truths = []
    for i in range(len(counts)):
        got = TRACKED_AFTER if i in full else alloc.get(i, 0.0)
        truths.append(round(100.0 * got / TRACKED_AFTER, 2))
    return [t['id'] for t in case['trophies']], truths


def backtest_practice(cases, params=None, verbose=True):
    out = []
    for case in cases:
        ids = [t['id'] for t in case['trophies']]
        rows = model_rows(case, params)
        base = baseline_rows(case)
        for kind in ('flat', 'lr'):
            _ids, truths = practice_truth(case, kind)
            cand = score(rows, truths, ids)
            ref = score(base, truths, ids)
            out.append({'case': case.get('id'), 'truth': kind, 'model': cand['loss'],
                        'base': ref['loss'], 'miss': cand['miss'],
                        'coverage': cand['coverage'], 'levels': cand['levels']})
            if verbose:
                print(line('%s/%s' % (case.get('id'), kind), cand, ref))
    return out


# ----------------------------------------------------------------------- 3. synthetic

class World:
    """A post-launch world with saturating trophy curves and plateaus.

    A trophy's rate follows C / (1 + exp(-(a + g * ln(1 + hours)))), calibrated to the
    seed rate at launch, so it rises towards its ceiling C and decelerates.  Tracked
    players arrive in a step at the documented republish and then grow towards a ceiling.
    The page republishes every few hours; reads inside a plateau are identical, and the
    printed rates carry the page's two decimals.
    """

    def __init__(self, seed, trophies, seed_rates):
        rng = random.Random(seed)
        self.trophies = trophies
        self.n = len(trophies)
        self.step_h = 4.55 + rng.uniform(-0.2, 0.2)
        self.cap_tracked = math.exp(rng.uniform(math.log(6e4), math.log(5e5)))
        self.tau_tracked = rng.uniform(30.0, 80.0)
        self.refresh = rng.choice((2.5, 3.5, 5.0))
        self.ceiling = []
        self.gamma = []
        self.alpha = []
        for i, trophy in enumerate(trophies):
            rate = seed_rates[i] if i < len(seed_rates) and seed_rates[i] is not None else 1.0
            tier = trophy.get('tier')
            cap = {'bronze': 55.0, 'silver': 70.0, 'gold': 85.0, 'platinum': 99.0}.get(tier, 55.0)
            cap = min(98.0, cap * math.exp(rng.uniform(-0.35, 0.35)))
            gamma = math.exp(rng.uniform(math.log(0.55), math.log(1.25)))
            if rate >= 99.5:
                cap, gamma = 100.0, 0.4
            self.ceiling.append(cap)
            self.gamma.append(gamma)
            frac = min(0.98, max(0.002, rate / 100.0))
            self.alpha.append(math.log(frac / (1.0 - frac)) - gamma * math.log(2.5))
        self._page = {}

    def tracked(self, hours):
        if hours < self.step_h:
            return TRACKED_BEFORE
        return TRACKED_AFTER + (self.cap_tracked - TRACKED_AFTER) * (
            1.0 - math.exp(-(hours - self.step_h) / self.tau_tracked))

    def rates(self, hours):
        out = []
        for i in range(self.n):
            z = self.alpha[i] + self.gamma[i] * math.log1p(max(hours, 0.0))
            out.append(self.ceiling[i] / (1.0 + math.exp(-z)))
        return out

    def stamp(self, hours):
        """The republish a read at `hours` shows: the page is static between them."""
        if hours < self.step_h:
            return 0.0
        return self.step_h + math.floor((hours - self.step_h) / self.refresh) * self.refresh

    def page(self, hours):
        stamp = self.stamp(hours)
        key = round(stamp, 6)
        if key not in self._page:
            self._page[key] = [round(r, 2) for r in self.rates(stamp)]
        return list(self._page[key])

    def case(self, checkpoint_h, horizon_h, cadence_min, case_id='syn'):
        hist = []
        h = -0.17
        while h < checkpoint_h - 1e-9:
            rates = self.page(h)
            total = self.tracked(self.stamp(h))
            stamp = iso(h)
            hist.append({'slot': stamp, 'at': stamp,
                         'players_tracked': int(round(total)),
                         'trophies_earned': int(round(sum(rates) * total / 100.0)),
                         'rates': rates})
            h += cadence_min / 60.0
        return {'schema': 'makefun/gta6-trophies-case/v1', 'id': case_id,
                'at': iso(checkpoint_h),
                'target': iso(checkpoint_h + horizon_h),
                'horizon_minutes': horizon_h * 60.0, 'levels': list(LEVELS),
                'game': {'name': 'Grand Theft Auto VI', 'platform': 'PS5',
                         'release_date': '2026-11-19',
                         'source': 'https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/'},
                'trophies': self.trophies, 'history': hist, 'notes': 'synthetic'}


def backtest_synthetic(trophies, seed_rates, worlds=12, seed=7, params=None, verbose=True):
    ids = [t['id'] for t in trophies]
    cand, base = [], []
    for w in range(worlds):
        world = World(seed + w, trophies, seed_rates)
        checkpoint = 59.5 + (w % 5) * 1.3
        case = world.case(checkpoint, 108.5, 10.0)
        truth = world.page(checkpoint + 108.5)
        cand.append(score(model_rows(case, params), truth, ids))
        base.append(score(baseline_rows(case), truth, ids))
    out = {'loss': mean(cand, 'loss'), 'model': mean(cand, 'loss'), 'base': mean(base, 'loss'),
           'miss': mean(cand, 'miss'), 'coverage': mean(cand, 'coverage'),
           'levels': [avg([c['levels'][k] for c in cand]) for k in range(5)],
           'worlds': worlds}
    if verbose:
        print(line('4.5 days', out, {'loss': out['base']}))
    return out


# --------------------------------------------------------------------------- 4. sweep

def sweep(cases, trophies, seed_rates, worlds=8, seed=7):
    print('  -- launch sigma (practice reconstruction) --')
    ids = {c.get('id'): [t['id'] for t in c['trophies']] for c in cases}
    for sigma in (0.04, 0.055, 0.07, 0.09, 0.12):
        vals = []
        for case in cases:
            rows = model_rows(case, {'launch_sigma': sigma})
            for kind in ('flat', 'lr'):
                _ids, truths = practice_truth(case, kind)
                vals.append(score(rows, truths, ids[case.get('id')])['loss'])
        print('     sigma=%.3f  mean loss %.4f' % (sigma, sum(vals) / len(vals)))
    print('  -- trend mix (synthetic, 4.5-day target) --')
    for mix in (0.30, 0.40, 0.45, 0.55):
        res = backtest_synthetic(trophies, seed_rates, worlds, seed,
                                 {'trend_mix': mix}, verbose=False)
        print('     mix=%.2f    loss %.4f  cov %.3f' % (mix, res['model'], res['coverage']))


def run_all(args):
    cases = load_cases(args.input)
    print('== truncation folds (flat truth) ==')
    trunc = backtest_truncations(cases)
    if trunc:
        print('  aggregate model %.4f  base %.4f'
              % (mean(trunc, 'model'), mean(trunc, 'base')))
    print('== practice reconstruction (refreshed page) ==')
    backtest_practice(cases)
    print('== synthetic post-launch worlds (4.5-day target) ==')
    seed_rates = (cases[0].get('history') or [{}])[-1].get('rates')
    backtest_synthetic(cases[0]['trophies'], seed_rates, args.worlds, args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='?', default='all',
                    choices=('all', 'truncations', 'practice', 'synthetic', 'sweep'))
    ap.add_argument('--input', default=None)
    ap.add_argument('--worlds', type=int, default=12)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    if args.what == 'all':
        run_all(args)
        return
    cases = load_cases(args.input)
    if args.what == 'truncations':
        backtest_truncations(cases)
    elif args.what == 'practice':
        backtest_practice(cases)
    elif args.what == 'synthetic':
        seed_rates = (cases[0].get('history') or [{}])[-1].get('rates')
        backtest_synthetic(cases[0]['trophies'], seed_rates, args.worlds, args.seed)
    else:
        seed_rates = (cases[0].get('history') or [{}])[-1].get('rates')
        sweep(cases, cases[0]['trophies'], seed_rates, args.worlds, args.seed)


if __name__ == '__main__':
    main()
