"""model.py -- the forecast behind main.py for gta6-trophy-rarity-forecast-01.

What the data is: the house reads Exophase's trophy page for GTA VI every 30 minutes
(then every 10), and Exophase republishes a game's counts on its own schedule, hours
apart.  So a case's history is a staircase: long runs of identical reads (one plateau
per Exophase refresh), and a rate only moves when the page is refreshed.  A trophy's
rate is 100 * earned / players_tracked, so a refresh moves every trophy at once and the
moves are shared.

How the forecast is made, per trophy:

  * the printed 2-decimal rate is snapped back to an integer count of earned players on
    that read's base (rate * tracked / 100, rounded), which removes the printing noise;
  * the reads are collapsed to one point per plateau, so a 250-read case fits in tens of
    points and every refresh counts the same;
  * a small census of curves is fitted to logit(rate) against ln(1 + hours since
    release): persistence, a straight line over the whole history, the same line over the
    last 24 hours, an approach to an asymptote, and the log-time line fitted as
    log(earned) - log(tracked) (the two counts the rate is made of, one of which is shared
    by every trophy);
  * the curve is chosen per case by a rolling-origin backtest on the case's own history,
    scored with the competition's own pinball loss, over gaps that are as long as the
    history allows (a family that looks fine three hours ahead can still be nonsense a
    week ahead);
  * the forecast is the LAST READ plus the curve's predicted *change* from the last
    read's time to the target, so the level always comes from the best evidence and only
    the drift from the fit; that change is shrunk towards the median change of the
    trophy's tier (a trophy with two or three refreshes carries almost no slope of its
    own), capped at a logit and a half, and then halved towards "nothing moves" -- the
    cheapest variance reduction there is, and the one the backtests score best;
  * the spread is the curve's prediction standard error, an extra random-walk term that
    grows with the horizon, mixed with the tier's median spread, all on the logit scale;
  * one prior, from the documented launch night: while the page has never been seen to
    refresh and the plateau is hours old, the next refresh is close by and it brings a
    much larger tracked base, so the rate is about to fall by a factor of ~4.5.

Everything is deterministic: no randomness, no dependence on dict order, no clock.
"""

import math
from datetime import datetime, timezone

# The levels the contest scores, in the order a row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# The standard normal's quantile at each level.
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Tunables; tools/backtest.py measures them and README.md reports the numbers.
PARAMS = {
    # Multiplier on the fitted curve's prediction standard error.
    "infl": 1.00,
    # Extra variance per day of horizon, in logit units squared per day.
    "rw_per_day": 0.34,
    # Floors and caps on a trophy's sigma, in logit units.
    "sig_min": 0.22,
    "sig_max": 2.60,
    # Share of a trophy's sigma taken from its tier's median sigma.
    "tier_mix": 0.35,
    # Shrinkage of a trophy's forecast change towards its tier's median change:
    # weight = n / (n + change_k) on the trophy's own change.
    "change_k": 3.0,
    # The fixed sigma the family census scores its pseudo-forecasts with.
    "select_sigma": 0.70,
    # A curve needs this many distinct plateaus to be fitted at all.
    "min_points": 2,
    # A fitted residual sd is floored here, so two points cannot claim a perfect fit.
    "resid_floor": 0.10,
    # A fitted change is capped here, in logit units, so no curve can run away on a
    # long extrapolation (a curve that wants 5 logits is telling us it is mis-specified).
    "delta_cap": 1.80,
    # The forecast change is this share of the fitted change: the other share is the
    # "nothing moves" forecast.  Halving it is the cheapest variance reduction there is
    # (tools/backtest.py scores it), because a trend that is right costs half a logit
    # and a trend that is wrong costs half as much as it would have.
    "mix_persist": 0.50,
    # Launch-night refresh prior (see _launch_prior).
    "prior_shift": 1.51,
    "prior_sd": 0.40,
    "prior_min_age": 3.0,
    "prior_ramp": 1.5,
    "prior_max_hours": 26.0,
    "prior_max_tracked": 6000.0,
    # A rate this close to 100 has no room to move; give it a hedged row instead.
    "full_pct": 99.5,
}

# The asymptote timescales the saturating candidates try, in hours.
SAT_TAUS = (12.0, 48.0)

# The census: every curve family the rolling-origin backtest chooses between.
CANDIDATES = (
    {"name": "persist", "type": "rate", "kind": "const"},
    {"name": "trend", "type": "rate", "kind": "log"},
    {"name": "trend24", "type": "rate", "kind": "log", "last24": True},
    {"name": "decomp", "type": "decomp", "kind": "log"},
) + tuple(
    {"name": "sat%g" % tau, "type": "rate", "kind": "sat", "tau": tau}
    for tau in SAT_TAUS
)


# ------------------------------------------------------------------------ scalar maths

def clamp_pct(value):
    """An earn rate in percent, clamped to where logit() is defined."""
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
    """Round half up, exactly as the house baseline's Math.round does."""
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
    """ISO 8601 UTC -> epoch seconds (a bare Z and fractional seconds both parse)."""
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


# ------------------------------------------------------------------------ curve maths

def _u(kind, hours, tau):
    """The regressor of a curve family at an age in hours since release."""
    h = hours if hours > 0.0 else 0.0
    if kind == "const":
        return 0.0
    if kind == "log":
        return math.log1p(h)
    if kind == "sat":
        return -math.exp(-h / tau)
    raise ValueError("unknown curve kind %r" % (kind,))


def _ols(points, kind, tau):
    """Least squares of y on _u(kind, h) at the given points.  None if degenerate.

    A constant curve is defined on a single point (that is the persistence candidate);
    every sloped curve needs two points to have a slope at all.
    """
    n = len(points)
    if n < (1 if kind == "const" else 2):
        return None
    ys = [y for _h, y in points]
    my = sum(ys) / n
    if kind == "const":
        sse = 0.0
        for y in ys:
            d = y - my
            sse += d * d
        return {"kind": kind, "tau": tau, "a": my, "b": 0.0, "mu": 0.0,
                "n": n, "sxx": 0.0, "sse": sse}
    us = [_u(kind, h, tau) for h, _y in points]
    mu = sum(us) / n
    sxx = sxy = 0.0
    for k in range(n):
        du = us[k] - mu
        sxx += du * du
        sxy += du * (ys[k] - my)
    if sxx <= 1e-12:
        return None
    b = sxy / sxx
    a = my - b * mu
    sse = 0.0
    for k in range(n):
        d = ys[k] - (a + b * us[k])
        sse += d * d
    return {"kind": kind, "tau": tau, "a": a, "b": b, "mu": mu,
            "n": n, "sxx": sxx, "sse": sse}


def _pred(fit, hours):
    return fit["a"] + fit["b"] * _u(fit["kind"], hours, fit["tau"])


def _pred_se(fit, points, hours, floor):
    """The curve's prediction standard error at an age, with a floor."""
    n = max(fit["n"], 2)
    resid = math.sqrt(fit["sse"] / max(n - 2, 1)) if fit["kind"] != "const" else 0.0
    resid = max(resid, floor)
    if fit["kind"] == "const":
        return resid
    u = _u(fit["kind"], hours, fit["tau"])
    lever = 1.0 + 1.0 / n + ((u - fit["mu"]) ** 2) / max(fit["sxx"], 1e-12)
    return resid * math.sqrt(lever)


# ------------------------------------------------------------------------ the model

class Forecaster:
    """One case in, one list of rows out: [{"id": n, "q": [q05..q95]}, ...]."""

    def __init__(self, case, params=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.case = case
        self.trophies = list(case.get("trophies") or [])
        self.n = len(self.trophies)
        self.tiers = [str(t.get("tier") or "bronze") for t in self.trophies]
        self.origin = self._origin_epoch()
        target = case.get("target")
        try:
            self.target_h = (parse_ts(target) - self.origin) / 3600.0
        except (TypeError, ValueError):
            self.target_h = float(case.get("horizon_minutes") or 0.0) / 60.0
        try:
            self.horizon_days = max(float(case.get("horizon_minutes")), 0.0) / 1440.0
        except (TypeError, ValueError):
            self.horizon_days = max(self.target_h, 0.0) / 24.0

        # read times, and per trophy the (hours, logit rate) and (hours, log count)
        # series, both raw (for the anchor) and collapsed to one point per plateau.
        self.read_h = []
        self.raw = [[] for _ in range(self.n)]
        self.counts = [[] for _ in range(self.n)]
        self.plateau = [[] for _ in range(self.n)]
        self.plateau_counts = [[] for _ in range(self.n)]
        self.tracked = []
        self._read_moves = False
        self._max_tracked = 0.0
        self._first_plateau_h = None
        self._build()

        self.family = "persist"
        self.family_scores = {}

    # -- input -------------------------------------------------------------

    def _origin_epoch(self):
        """Hours are counted from the release day's start; the first read if earlier."""
        stamps = []
        release = (self.case.get("game") or {}).get("release_date")
        if release:
            try:
                stamps.append(parse_ts(str(release) + "T00:00:00Z"))
            except (TypeError, ValueError):
                pass
        hist = self.case.get("history") or []
        if hist:
            try:
                stamps.append(parse_ts(hist[0].get("at") or hist[0].get("slot")))
            except (TypeError, ValueError):
                pass
        return min(stamps) if stamps else 0.0

    def _build(self):
        prev = None
        first_h = None
        for read in self.case.get("history") or []:
            stamp = read.get("at") or read.get("slot")
            try:
                h = (parse_ts(stamp) - self.origin) / 3600.0
            except (TypeError, ValueError):
                continue
            base = _as_number(read.get("players_tracked"))
            if base is not None and base > self._max_tracked:
                self._max_tracked = base
            self.read_h.append(h)
            self.tracked.append((h, base))
            rates = read.get("rates") or []
            snapshot = []
            for i in range(self.n):
                rate = rates[i] if i < len(rates) else None
                if rate is None:
                    continue
                pct, count = _snap(rate, base)
                snapshot.append(pct)
                self.raw[i].append((h, logit(pct)))
                self.counts[i].append((h, math.log(count + 0.5)))
            if prev is not None and snapshot != prev:
                self._read_moves = True
                first_h = h
            prev = snapshot
        self.read_h.sort()
        self.tracked.sort(key=lambda pt: pt[0])
        self._first_plateau_h = first_h if first_h is not None else (
            self.read_h[0] if self.read_h else None)
        for i in range(self.n):
            self.raw[i].sort(key=lambda pt: pt[0])
            self.counts[i].sort(key=lambda pt: pt[0])
            self.plateau[i] = _collapse(self.raw[i])
            self.plateau_counts[i] = _collapse(self.counts[i])
        self.tracked_collapsed = _collapse([(h, b) for h, b in self.tracked
                                            if b is not None and b > 0])

    # -- fitting one candidate for one trophy ------------------------------

    def _series(self, cand, points):
        pts = points
        if cand.get("last24") and pts:
            cut = pts[-1][0] - 24.0
            sub = [pt for pt in pts if pt[0] >= cut]
            if len(sub) >= 3:
                pts = sub
        return pts

    def _fit_trophy(self, cand, i, limit=None):
        """Fit a candidate for trophy i, optionally on the history up to `limit` hours.

        Returns an object the evaluator understands, or None when the data cannot carry
        the curve (too few points, a degenerate regressor, no tracked series).
        """
        pts = self.plateau[i] if limit is None else _upto(self.plateau[i], limit)
        if cand.get("type") == "decomp":
            cp = self.plateau_counts[i] if limit is None else _upto(self.plateau_counts[i],
                                                                   limit)
            tp = self.tracked_collapsed if limit is None else _upto(self.tracked_collapsed,
                                                                    limit)
            cp = self._series(cand, cp)
            if len(cp) < self.p["min_points"] or len(tp) < self.p["min_points"]:
                return None
            lp = [(h, math.log(b)) for h, b in tp]
            f_count = _ols(cp, cand["kind"], cand.get("tau"))
            f_base = _ols(lp, cand["kind"], cand.get("tau"))
            if f_count is None or f_base is None:
                return None
            return {"mode": "decomp", "fc": f_count, "fb": f_base, "cp": cp, "lp": lp}
        pts = self._series(cand, pts)
        need = 1 if cand["kind"] == "const" else self.p["min_points"]
        if len(pts) < need:
            return None
        fit = _ols(pts, cand["kind"], cand.get("tau"))
        if fit is None:
            return None
        return {"mode": "rate", "fit": fit, "pts": pts}

    def _eval(self, obj, hours):
        """(centre in percent, prediction standard error) of a fitted candidate."""
        floor = self.p["resid_floor"]
        if obj["mode"] == "rate":
            centre = inv_logit(_pred(obj["fit"], hours))
            return centre, _pred_se(obj["fit"], obj["pts"], hours, floor)
        # the rate is 100 * earned / tracked: fit both counts in log space
        log_rate = _pred(obj["fc"], hours) - _pred(obj["fb"], hours)
        log_rate = min(4.6, max(-9.2, log_rate))
        se = math.hypot(_pred_se(obj["fc"], obj["cp"], hours, floor),
                        _pred_se(obj["fb"], obj["lp"], hours, floor))
        return 100.0 * math.exp(log_rate), se

    # -- choosing the curve on the case's own history ----------------------

    def _origins(self):
        """Up to three pseudo-checkpoints (train end, evaluation time).

        The gaps are deliberately long: a family that looks fine three hours ahead can
        still be nonsense a week ahead, so the census is judged on the longest gaps its
        own history allows.
        """
        hs = sorted(set(self.read_h))
        if len(hs) < 4:
            return []
        span = hs[-1] - hs[0]
        if span <= 0.0:
            return []
        out = []
        for frac, gap in ((0.40, 0.30), (0.50, 0.35), (0.60, 0.40)):
            cut = hs[0] + frac * span
            train = cut - gap * span
            if train <= hs[0]:
                continue
            later = [h for h in hs if h >= cut]
            if not later:
                continue
            out.append((train, later[0]))
        return out

    def _select(self):
        origins = self._origins()
        sigma = self.p["select_sigma"]
        prepared = []
        for train, evaluate in origins:
            truths = [_first_at_or_after(self.raw[i], evaluate) for i in range(self.n)]
            prepared.append((train, evaluate, truths))
        best, best_score = CANDIDATES[0]["name"], float("inf")
        for cand in CANDIDATES:
            total = 0.0
            items = 0
            for train, evaluate, truths in prepared:
                for i in range(self.n):
                    truth = truths[i]
                    if truth is None:
                        continue
                    obj = self._fit_trophy(cand, i, train)
                    if obj is None:
                        obj = self._fit_trophy(CANDIDATES[0], i, train)
                        if obj is None:
                            continue
                    centre, _se = self._eval(obj, evaluate)
                    zc = logit(centre)
                    for tau, zq in zip(LEVELS, Z):
                        d = truth - (zc + zq * sigma)
                        total += 2.0 * max(tau * d, (tau - 1.0) * d)
                        items += 1
            score = total / items if items else float("inf")
            self.family_scores[cand["name"]] = score
            if score < best_score - 1e-12:
                best, best_score = cand["name"], score
        self.family = best
        for cand in CANDIDATES:
            if cand["name"] == best:
                return cand
        return CANDIDATES[0]

    # -- the launch-night refresh prior -----------------------------------

    def _launch_prior(self):
        """Probability that the page refreshes before the target, and the shift it brings.

        Documented launch night: the page stood unchanged from 2026-11-18T23:48Z through
        the 04:00Z read, and the next read (04:30Z) showed 3,868 players tracked instead
        of 638 and 4,276 trophies earned instead of 943.  The game's one trophy at 100%
        accounts for the whole base, so the other 54 went from 305 earned to 408 -- their
        rates fell by 638/3868 * 408/305 = 4.5, or -1.51 in logit space.  The prior is
        armed only while the page has not been seen to refresh, the plateau is a few
        hours old (a page younger than three hours had never been seen to move), and the
        read is still on launch night.
        """
        p = self.p
        if self._read_moves or not self.read_h:
            return 0.0, 0.0
        if self.read_h[-1] > p["prior_max_hours"]:
            return 0.0, 0.0
        if self._max_tracked > p["prior_max_tracked"]:
            return 0.0, 0.0
        start = self._first_plateau_h
        if start is None:
            return 0.0, 0.0
        age = self.target_h - start
        prob = min(1.0, max(0.0, (age - p["prior_min_age"]) / p["prior_ramp"]))
        return prob, p["prior_shift"]

    # -- output ------------------------------------------------------------

    def forecast(self):
        p = self.p
        cand = self._select()
        days = max(self.horizon_days, 0.0)
        rw_var = (p["rw_per_day"] ** 2) * days

        anchors, deltas, sigmas = [], [], []
        for i in range(self.n):
            series = self.raw[i]
            if series:
                last_h, anchor = series[-1]
            else:
                last_h = self.read_h[-1] if self.read_h else 0.0
                anchor = logit(0.01)
            anchors.append(anchor)
            obj = self._fit_trophy(cand, i)
            if obj is None:
                deltas.append(0.0)
                sigmas.append(max(p["sig_min"], math.sqrt(rw_var)))
                continue
            centre, se = self._eval(obj, self.target_h)
            # the change is the curve's own move from the last read's time to the target,
            # so a persistence fit changes exactly nothing and the level stays the last read
            start, _se = self._eval(obj, last_h)
            delta = logit(centre) - logit(start)
            if delta > p["delta_cap"]:
                delta = p["delta_cap"]
            elif delta < -p["delta_cap"]:
                delta = -p["delta_cap"]
            deltas.append(delta)
            sigmas.append(math.sqrt((p["infl"] * se) ** 2 + rw_var))

        tier_delta = _median_by(self.tiers, deltas)
        tier_sigma = _median_by(self.tiers, sigmas)

        centres = []
        for i in range(self.n):
            npts = len(self._series(cand, self.plateau[i])) if self.plateau[i] else 0
            w = npts / (npts + p["change_k"]) if npts else 0.0
            own = deltas[i]
            prior = tier_delta.get(self.tiers[i], 0.0)
            centres.append(anchors[i] + p["mix_persist"] * (w * own + (1.0 - w) * prior))

        prob, shift = self._launch_prior()
        mix = p["tier_mix"]
        rows = []
        for i in range(self.n):
            s = sigmas[i]
            ts = tier_sigma.get(self.tiers[i], s)
            s = math.sqrt((1.0 - mix) * s * s + mix * ts * ts)
            mu = centres[i]
            if prob > 0.0:
                # two outcomes: the plateau stands (0) or the page refreshes (-shift)
                mu -= prob * shift
                s = math.sqrt(s * s + prob * (p["prior_sd"] ** 2)
                              + prob * (1.0 - prob) * (shift ** 2))
            s = min(p["sig_max"], max(p["sig_min"], s))
            q = [half_up(inv_logit(mu + zq * s)) for zq in Z]
            q = [min(100.0, max(0.0, float(v))) for v in q]
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
            rows.append({"id": self.trophies[i].get("id", i + 1), "q": self._hedge(q, i)})
        return rows

    def _hedge(self, q, i):
        """A trophy at 100% cannot rise and is reported as 100: keep a low tail anyway."""
        series = self.raw[i]
        if not series:
            return q
        last = inv_logit(series[-1][1])
        if last < self.p["full_pct"]:
            return q
        return [min(97.0, q[0]), min(99.5, q[1]), max(q[1], 99.5), 100.0, 100.0]


# ------------------------------------------------------------------------ helpers

def _as_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _snap(rate, base):
    """A printed rate -> (de-noised percent, integer count of earned players)."""
    raw = _as_number(rate)
    if raw is None:
        return 0.01, 0.0
    if base is not None and base > 0:
        count = int(round(raw * base / 100.0))
        if count <= 0:
            return 0.01, 0.0
        if count >= base:
            return 99.99, base
        return 100.0 * count / base, float(count)
    return clamp_pct(raw), max(1.0, raw)


def _collapse(points):
    """One point per plateau: keep the last read of each run of equal values."""
    out = []
    for pt in points:
        if out and abs(pt[1] - out[-1][1]) <= 1e-12:
            out[-1] = pt
        else:
            out.append(pt)
    return out


def _upto(points, limit):
    return [pt for pt in points if pt[0] <= limit + 1e-9]


def _first_at_or_after(points, hours):
    for h, y in points:
        if h >= hours - 1e-9:
            return y
    return None


def _median_by(keys, values):
    groups = {}
    for key, value in zip(keys, values):
        groups.setdefault(key, []).append(value)
    return {key: median(vals) for key, vals in groups.items()}


def forecast_case(case, params=None):
    """A case dict -> {"trophies": [{"id": n, "q": [...]}, ...]}."""
    return {"trophies": Forecaster(case, params).forecast()}
