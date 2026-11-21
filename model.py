"""model.py -- the forecast behind main.py for gta6-trophy-rarity-forecast-01.

The page Exophase prints is a staircase.  It republishes a game's counts every few
hours, so a case's history is long runs of identical reads (plateaus) broken by a few
steps.  A trophy's rate is 100 * earned / players_tracked, printed to two decimals.
Two consequences decide this model:

  * On a plateau the rate cannot move until the next republish, so a target that lands
    inside one is the last read, and a model that invents a trend there pays for it.
  * A republish is not a smooth function of the last read: the new counts are whole
    numbers of players, and which trophies the new players earned is not proportional
    to the old ones.  The level of a republished page is predictable; the per-trophy
    deviation from it is not, and it is large.  Round 2 forecast the level and priced
    the deviation at 0.055 logits, and the board answered with 49% coverage.  So the
    band here is wide on purpose: a tail crossed at the 5% level costs 1.9 per logit
    unit, while a band that is too wide costs 0.1 per unit.  Honest beats confident.

Three regimes, chosen from the case itself:

  launch   The page has never moved, the base is still the launch-night 638 tracked and
           the summed earned counts are the documented 943.  The house published what
           the first republish did: 3,868 tracked, 4,276 earned, every rate but the one
           at 100% lower.  The 100% trophy holds the whole old and new base, so the
           other trophies' counts grew from 305 to 408 while the base grew from 638 to
           3,868: a trophy's share of the 408 is what decides its new rate.  The level is
           a per-trophy transform of the old count, not a uniform logit shift (a shift is
           wrong for the common trophies); the band around it is the calibrated
           dispersion of that 408-way split, plus the integer rounding of small counts.

  plateau  Reads have not moved but the launch transform does not describe the page.
           The last read is the forecast; the band widens slowly with the horizon,
           because the only thing that can move a rate is a republish we cannot see.

  trend    Reads have moved.  Fit logit(rate) against log time over the last day (or the
           whole history when shorter), extrapolate to the target, cap the move, shrink
           each trophy's move toward its tier's median, keep a fraction of it, and
           shrink it further when the newest reads are a fresh plateau -- a trend whose
           last step is hours old is not a trend that is still running.

Uncertainty is a logit-scale standard deviation per trophy, so the five quantiles are
mu + z * sigma with z = -1.645, -0.674, 0, 0.674, 1.645.  The dispersion constants are
fitted by tools/backtest.py to the board's measurement of how far a real refreshed page
sits from the proportional guess; they are not chosen by taste.

Everything is deterministic: no randomness, no clock, no dependence on dict order.
"""

import math
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

PARAMS = {
    # -- the documented launch-night republish --------------------------------
    "launch_tracked": 638.0,
    "launch_earned": 943.0,
    "launch_after_tracked": 3868.0,
    "launch_after_earned": 4276.0,
    "launch_tracked_tol": 25.0,
    "launch_earned_tol": 12.0,
    # The republish landed between the 04:00Z and 04:30Z reads: ramp the probability
    # from 0 at 04:00 to 1 at 04:30, hours counted from release midnight.
    "launch_ramp_from_h": 4.02,
    "launch_ramp_to_h": 4.45,
    # Logit dispersion of one trophy's share of the republished earned counts.  Fitted
    # by tools/backtest.py calibrate: this is the s at which round 2's 0.055-logit band
    # reproduces the board's 0.4909 coverage / 0.159 loss / 0.05 pp miss.
    "launch_disp": 0.36,
    # A small count is rounded to a whole player: half a player is a big share of it.
    "launch_count_jitter": 0.35,
    # A target well past the documented republish may see another one; widen for it.
    "launch_late_per_sqrt_h": 0.10,
    "launch_sigma_min": 0.05,
    "launch_sigma_max": 1.30,
    # -- a plateau with no documented transform -------------------------------
    "plateau_sigma": 0.06,
    "plateau_grow": 0.012,
    # A plateau that is not the documented launch snapshot is a page that WILL refresh
    # somewhere inside the horizon, and the step it takes is not small.  Widening with
    # the horizon is what keeps a short-history trial inside its band.
    "plateau_grow_wild": 0.09,
    "plateau_sigma_cap": 1.10,
    # -- ordinary moving history ---------------------------------------------
    "trend_kind": "log",
    "trend_mix": 0.80,
    "trend_cap": 1.40,
    "tier_shrink_k": 3.5,
    # A trend whose newest reads are a plateau is stale: halve it for every this many
    # hours the page has held still.
    "plateau_hold_h": 20.0,
    "plateau_hold_floor": 0.35,
    "base_sigma": 0.12,
    "rw_per_sqrt_day": 0.07,
    # Extrapolation error grows with the distance moved, not just with the horizon.
    "trend_sigma_scale": 0.15,
    "sig_min": 0.10,
    "sig_max": 1.60,
    # A printed 99.5%+ rate is the one at 100%: it has no room to rise.
    "full_pct": 99.5,
    "full_sigma": 0.02,
    # Round the emitter's numbers to the page's own precision where it pays.
    "round_places": 4,
    "median_places": 2,
}


# ------------------------------------------------------------------------ scalar maths

def clamp_pct(value):
    """An earn rate in percent, clamped to the interval where logit() is defined."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.01
    if not math.isfinite(v):
        return 0.01
    return min(99.99, max(0.01, v))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(z):
    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0
    return 100.0 / (1.0 + math.exp(-z))


def half_up(value, places=4):
    """Round half up, as the house baseline's Math.round does."""
    scale = 10 ** places
    scaled = value * scale
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / scale
    return int(out) if float(out).is_integer() else out


def median(values):
    vals = sorted(values)
    n = len(vals)
    if not n:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def parse_ts(text):
    """ISO 8601 UTC -> epoch seconds (Z and fractional seconds both parse)."""
    s = str(text).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _median_by(keys, values):
    groups = {}
    for key, value in zip(keys, values):
        groups.setdefault(key, []).append(value)
    return {key: median(vals) for key, vals in groups.items()}


# -------------------------------------------------------------------------- the model

class Forecaster:
    """One case in, one list of rows out: [{"id": n, "q": [q05..q95]}, ...]."""

    def __init__(self, case, params=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.case = case if isinstance(case, dict) else {}
        self.trophies = list(self.case.get("trophies") or [])
        self.n = len(self.trophies)
        self.tiers = [str(t.get("tier") or "bronze") for t in self.trophies]
        self.origin = self._origin()
        self.target_h = self._target_hours()

        # Per trophy: points of (hours, de-noised percent, whole earned count).
        self.series = [[] for _ in range(self.n)]
        self.read_h = []
        self.tracked = []
        self.moved = False
        self.max_tracked = 0.0
        self.last_h = 0.0
        self.earned_last = None
        self._build()
        self.horizon_days = max(self.target_h - self.last_h, 0.0) / 24.0

    # -- input -------------------------------------------------------------

    def _origin(self):
        """Hours count from the release day's midnight, else from the first read."""
        release = (self.case.get("game") or {}).get("release_date")
        if release:
            try:
                return parse_ts(str(release) + "T00:00:00Z")
            except (TypeError, ValueError):
                pass
        for read in self.case.get("history") or []:
            try:
                return parse_ts(read.get("at") or read.get("slot"))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _target_hours(self):
        try:
            return (parse_ts(self.case.get("target")) - self.origin) / 3600.0
        except (TypeError, ValueError):
            try:
                return max(float(self.case.get("horizon_minutes")), 0.0) / 60.0
            except (TypeError, ValueError):
                return 0.0

    def _build(self):
        prev = None
        for read in self.case.get("history") or []:
            stamp = read.get("at") or read.get("slot")
            try:
                h = (parse_ts(stamp) - self.origin) / 3600.0
            except (TypeError, ValueError):
                continue
            base = _number(read.get("players_tracked"))
            if base is not None and base > self.max_tracked:
                self.max_tracked = base
            self.read_h.append(h)
            self.tracked.append((h, base))
            rates = read.get("rates") or []
            snapshot = []
            for i in range(self.n):
                rate = rates[i] if i < len(rates) else None
                if rate is None:
                    snapshot.append(None)
                    continue
                pct, count = self._snap(rate, base)
                snapshot.append(pct)
                self.series[i].append((h, pct, count))
            if prev is not None:
                # a null in either read is not evidence of a move
                if any(a is not None and b is not None and abs(a - b) > 1e-9
                       for a, b in zip(prev, snapshot)):
                    self.moved = True
            prev = snapshot
            self.last_h = h
            self.earned_last = self._earned(rate_row=rates, base=base)

        for i in range(self.n):
            self.series[i].sort(key=lambda pt: pt[0])
        self.read_h.sort()
        self.tracked.sort(key=lambda pt: pt[0])

    def _snap(self, rate, base):
        """A 2-decimal printed rate -> (count/base as percent, whole earned count)."""
        raw = _number(rate)
        if raw is None:
            return 0.01, None
        if base is not None and base > 0:
            count = int(round(raw * base / 100.0))
            if count <= 0:
                return 0.01, 0
            if count >= base:
                return 99.99, int(round(base))
            return 100.0 * count / base, count
        return clamp_pct(raw), None

    def _earned(self, rate_row, base):
        """The read's printed total earned, as the sum of the snapped counts."""
        if base is None:
            return None
        total = 0.0
        any_rate = False
        for i in range(self.n):
            rate = rate_row[i] if i < len(rate_row) else None
            if rate is None:
                continue
            any_rate = True
            _pct, count = self._snap(rate, base)
            total += count if count is not None else 0.0
        return total if any_rate else None

    # -- helpers -----------------------------------------------------------

    def _last(self, i):
        """(hours, logit, count) of trophy i's newest read, or None."""
        pts = self.series[i]
        if not pts:
            return None
        h, pct, count = pts[-1]
        return h, logit(pct), count

    def _plateaus(self, i):
        """Collapse consecutive equal readings to one point per plateau."""
        out = []
        for h, pct, _count in self.series[i]:
            lg = logit(pct)
            if out and abs(lg - out[-1][1]) <= 1e-9:
                out[-1] = (h, lg)
            else:
                out.append((h, lg))
        return out

    def _is_full(self, i):
        last = self._last(i)
        return last is not None and inv_logit(last[1]) >= self.p["full_pct"]

    def _still_hours(self, i):
        """How long trophy i's newest plateau has held, in hours."""
        pts = self.series[i]
        if len(pts) < 2:
            return 0.0
        newest = pts[-1][1]
        for h, pct, _c in reversed(pts[:-1]):
            if abs(pct - newest) > 1e-9:
                return max(pts[-1][0] - h, 0.0)
        return max(pts[-1][0] - pts[0][0], 0.0)

    # -- regime 1: the documented launch republish -------------------------

    def _launch_page(self):
        """True when the page is provably the launch-night snapshot (its rates are frozen)."""
        p = self.p
        if self.moved or not self.read_h or self.max_tracked <= 0:
            return False
        if abs(self.max_tracked - p["launch_tracked"]) > p["launch_tracked_tol"]:
            return False
        if self.earned_last is None:
            return False
        if abs(self.earned_last - p["launch_earned"]) > p["launch_earned_tol"]:
            return False
        return any(self._is_full(i) for i in range(self.n))

    def _launch_state(self):
        """The documented launch plateau, with the probability its republish has landed.

        Returns None unless the page is provably the launch-night snapshot: never
        moved, still ~638 tracked, still ~943 earned, and one trophy at 100%.
        """
        p = self.p
        if self.moved or not self.read_h or self.max_tracked <= 0:
            return None
        if abs(self.max_tracked - p["launch_tracked"]) > p["launch_tracked_tol"]:
            return None
        if self.earned_last is None:
            return None
        if abs(self.earned_last - p["launch_earned"]) > p["launch_earned_tol"]:
            return None
        if not any(self._is_full(i) for i in range(self.n)):
            return None
        lo = p["launch_ramp_from_h"]
        hi = p["launch_ramp_to_h"]
        prob = (self.target_h - lo) / (hi - lo) if hi > lo else 1.0
        return min(1.0, max(0.0, prob))

    def _launch_rows(self, prob):
        """Median: each trophy's share of the republished 408 earned counts.

        Band: the calibrated dispersion of that share.  The share is the only thing an
        unseen page can disagree with us about, and it is worth more than the level:
        at s = 0.30 the band spans about 0.49 logits either way, which is what an
        honest forecast of a 408-way split looks like.
        """
        p = self.p
        after_tracked = p["launch_after_tracked"]
        after_earned = p["launch_after_earned"]
        counts = {}
        full = []
        for i in range(self.n):
            if self._is_full(i):
                full.append(i)
                counts[i] = after_tracked
                continue
            last = self._last(i)
            counts[i] = last[2] if last is not None else None
        held_before = sum(after_tracked for _ in full)
        others_before = sum(c or 0.0 for c in counts.values()) - held_before
        others_after = after_earned - after_tracked * len(full)
        factor = others_after / others_before if others_before > 0.0 else 0.0

        rows = []
        for i in range(self.n):
            last = self._last(i)
            mu_last = last[1] if last is not None else logit(0.01)
            if i in full:
                mu_ref = logit(99.99)
                s_ref = p["full_sigma"]
            elif counts[i] is None or factor <= 0.0:
                mu_ref = mu_last
                s_ref = p["base_sigma"]
            else:
                raw = 100.0 * counts[i] * factor / after_tracked
                # the documented page had every rate but the 100% one lower
                if last is not None:
                    raw = min(raw, inv_logit(mu_last) * 0.999)
                mu_ref = logit(raw)
                # a split of whole counts: rounding half a player matters when the
                # count is small, and the split's own dispersion does the rest
                jitter = p["launch_count_jitter"] / max(counts[i], 1.0)
                s_ref = math.sqrt(p["launch_disp"] ** 2 + jitter ** 2)
            mu = (1.0 - prob) * mu_last + prob * mu_ref
            s = (1.0 - prob) * p["plateau_sigma"] + prob * s_ref
            if 0.0 < prob < 1.0:
                # the republish may or may not have landed: that alone is a big spread
                s = math.sqrt(s * s + prob * (1.0 - prob) * (mu_ref - mu_last) ** 2)
            late = max(self.target_h - p["launch_ramp_to_h"], 0.0)
            s = math.sqrt(s * s + (p["launch_late_per_sqrt_h"] ** 2) * late)
            s = min(p["launch_sigma_max"], max(p["launch_sigma_min"], s))
            rows.append((mu, s, i))
        return rows

    # -- regime 2: a plain plateau -----------------------------------------

    def _plateau_rows(self):
        p = self.p
        frozen = self._launch_page()
        rows = []
        for i in range(self.n):
            last = self._last(i)
            if last is None:
                rows.append((logit(0.01), p["plateau_sigma_cap"], i))
                continue
            h, mu, _count = last
            gap = max(self.target_h - h, 0.0)
            if frozen:
                # the house documented the page frozen from 23:48Z to 04:00Z: a target
                # inside that stretch cannot have moved, so the band stays narrow
                s = p["plateau_sigma"] + p["plateau_grow"] * math.sqrt(gap)
            else:
                s = math.sqrt(p["plateau_sigma"] ** 2
                              + (p["plateau_grow_wild"] ** 2) * gap)
            if self._is_full(i):
                s = p["full_sigma"]
            rows.append((mu, min(p["plateau_sigma_cap"], s), i))
        return rows

    # -- regime 3: an ordinary moving history ------------------------------

    def _trend_x(self, hours):
        """The regressor of the trend fit: log time, or linear hours."""
        if self.p["trend_kind"] == "log":
            return math.log1p(max(hours, 0.0))
        return hours

    def _fit_slope(self, points):
        """OLS slope of logit on the trend regressor, and the residual sd."""
        n = len(points)
        if n < 2:
            return None
        xs = [self._trend_x(pt[0]) for pt in points]
        mx = sum(xs) / n
        my = sum(pt[1] for pt in points) / n
        sxx = sxy = 0.0
        for x, pt in zip(xs, points):
            dx = x - mx
            sxx += dx * dx
            sxy += dx * (pt[1] - my)
        if sxx <= 1e-12:
            return None
        slope = sxy / sxx
        sse = 0.0
        for x, pt in zip(xs, points):
            d = pt[1] - (my + slope * (x - mx))
            sse += d * d
        resid = math.sqrt(sse / max(n - 2, 1)) if n >= 3 else 0.0
        return slope, resid, n

    def _trend_rows(self):
        p = self.p
        deltas, sigmas, stills = [], [], []
        for i in range(self.n):
            last = self._last(i)
            if last is None:
                deltas.append(0.0)
                sigmas.append(p["base_sigma"])
                stills.append(0.0)
                continue
            h_last, mu_last, _count = last
            if self._is_full(i):
                deltas.append(0.0)
                sigmas.append(p["full_sigma"])
                stills.append(0.0)
                continue
            pts = self._plateaus(i)
            window = [pt for pt in pts if pt[0] >= h_last - 24.0]
            if len(window) < 3:
                window = pts
            fit = self._fit_slope(window)
            if fit is None:
                deltas.append(0.0)
                sigmas.append(p["base_sigma"])
                stills.append(self._still_hours(i))
                continue
            slope, resid, _npts = fit
            gap = (self._trend_x(self.target_h) - self._trend_x(h_last)
                   if self.target_h > h_last else 0.0)
            delta = slope * gap
            deltas.append(min(p["trend_cap"], max(-p["trend_cap"], delta)))
            sigmas.append(math.sqrt(p["base_sigma"] ** 2 + resid ** 2))
            stills.append(self._still_hours(i))

        tier_delta = _median_by(self.tiers, deltas)
        rows = []
        for i in range(self.n):
            last = self._last(i)
            mu_last = last[1] if last is not None else logit(0.01)
            npts = len(self._plateaus(i))
            w = npts / (npts + p["tier_shrink_k"]) if npts else 0.0
            prior = tier_delta.get(self.tiers[i], 0.0)
            delta = p["trend_mix"] * (w * deltas[i] + (1.0 - w) * prior)
            # a page that has held still for hours is not still moving that fast
            hold = 1.0 / (1.0 + stills[i] / p["plateau_hold_h"])
            delta *= max(p["plateau_hold_floor"], hold)
            s = math.sqrt(sigmas[i] ** 2
                          + (p["rw_per_sqrt_day"] ** 2) * max(self.horizon_days, 0.0)
                          + (p["trend_sigma_scale"] * delta) ** 2)
            s = min(p["sig_max"], max(p["sig_min"], s))
            rows.append((mu_last + delta, s, i))
        return rows

    # -- output ------------------------------------------------------------

    def forecast(self):
        """Rows in the case's trophy order, one per trophy."""
        if not self.n:
            return []
        self.horizon_days = max(self.target_h - self.last_h, 0.0) / 24.0

        prob = self._launch_state()
        if prob is not None and prob > 0.0:
            rows = self._launch_rows(prob)
        elif not self.moved:
            rows = self._plateau_rows()
        else:
            rows = self._trend_rows()
        if rows is None:
            rows = self._plateau_rows()

        out = []
        places = self.p["round_places"]
        mplaces = self.p["median_places"]
        for mu, s, i in rows:
            s = min(self.p["sig_max"], max(1e-6, s))
            q = [half_up(inv_logit(mu + z * s), places) for z in Z]
            if mplaces is not None:
                # the page prints 2 decimals: the median may as well sit on that grid
                q[2] = max(0.01, min(100.0, float(half_up(q[2], mplaces))))
            q = [min(100.0, max(0.0, float(v))) for v in q]
            for k in range(1, len(q)):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
            out.append({"id": self.trophies[i].get("id", i + 1), "q": q})
        return out


def forecast_case(case, params=None):
    """A case dict -> {"trophies": [{"id": n, "q": [...]}, ...]}."""
    return {"trophies": Forecaster(case, params).forecast()}
