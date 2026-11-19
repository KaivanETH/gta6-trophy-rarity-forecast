"""main.py -- entrypoint for gta6-trophy-rarity-forecast-01 (Python 3.13, stdlib only).

The grader runs `python3 -P -s /bot/main.py` once per case, with only that case's file in
/input, and reads one forecast from /output/<id>.out:

    {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}

exactly one row per trophy of the case, each id once, five finite non-decreasing numbers
from 0 to 100, at most 65,536 bytes.  Anything else is an invalid case: it is scored as
if every quantile were that trophy's last read, and one invalid case costs the whole
entry.  So this file is deliberately boring: it loads, it asks model.forecast_case for
the numbers, it checks them against the contract, and it writes them the same way every
time.  If the model raises or returns something malformed for a case, this file writes
the house baseline's persistence forecast for it instead -- a wide, honest forecast is
worth far more than a crashed run.

Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and
/output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
import traceback
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds the
# model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402  (after the sys.path line, on purpose)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
MAX_BYTES = 65536


def persistence_rows(case):
    """The house baseline, number for number: last read + random walk on the logit."""
    hist = case.get("history") or []
    rates = (hist[-1].get("rates") if hist else None) or []
    days = max(_num(case.get("horizon_minutes"), 0.0), 0.0) / 1440.0
    spread = 0.5 * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        rate = rates[i] if i < len(rates) else None
        lg = model.logit(rate if rate is not None else 0.01)
        q = [model.half_up(model.inv_logit(lg + z * spread)) for z in Z]
        rows.append({"id": trophy.get("id", i + 1), "q": q})
    return rows


def _num(value, fallback):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def repair_rows(case, rows):
    """Check rows against the output contract and return them in trophy order, or None.

    The contract: one row per trophy, each id once, five finite numbers in 0..100 and
    non-decreasing.  A row with a broken shape is not repairable (the case would be
    invalid anyway), but a number that is merely out of order or out of range is fixed
    in place, because the contract allows any order of rows and any equal neighbours.
    """
    try:
        if not isinstance(rows, list):
            return None
        wanted = [t.get("id") for t in (case.get("trophies") or [])]
        if not wanted:
            return None
        by_id = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            key = row.get("id")
            values = row.get("q")
            if key is None or not isinstance(values, (list, tuple)) or len(values) != 5:
                return None
            if key in by_id:
                return None
            q = [_num(v, None) for v in values]
            if any(v is None for v in q):
                return None
            q = [min(100.0, max(0.0, v)) for v in q]
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
            by_id[key] = q
        if len(by_id) != len(wanted) or sorted(by_id) != sorted(wanted):
            return None
        return [{"id": tid, "q": by_id[tid]} for tid in wanted]
    except Exception:  # a malformed case is not worth a crash
        traceback.print_exc(file=sys.stderr)
        return None


def forecast_case(case, name):
    """The forecast for one case: the model's, or the baseline's if that cannot be used."""
    if not isinstance(case, dict):
        raise ValueError("%s is not a JSON object" % (name,))
    if case.get("levels") is not None and list(case["levels"]) != LEVELS:
        print("%s: levels %r, forecasting %r anyway" % (name, case.get("levels"), LEVELS),
              file=sys.stderr)
    try:
        rows = model.forecast_case(case).get("trophies")
        repaired = repair_rows(case, rows)
        if repaired is not None:
            return {"trophies": repaired}
        print("%s: model rows broke the contract, falling back" % (name,), file=sys.stderr)
    except Exception:  # a model bug must never cost the case
        traceback.print_exc(file=sys.stderr)
        print("%s: model raised, falling back" % (name,), file=sys.stderr)
    return {"trophies": repair_rows(case, persistence_rows(case)) or []}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(INPUT_DIR.glob("*.in"))
    if not paths:
        print("no *.in case files in %s" % (INPUT_DIR,), file=sys.stderr)
    for path in paths:
        # one bad file must not take the whole run down: the grader sends one case per
        # run, but a local run over a directory sends several
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            payload = json.dumps(forecast_case(case, path.name), separators=(",", ":"),
                                 allow_nan=False) + "\n"
            data = payload.encode("utf-8")
            if len(data) > MAX_BYTES:
                raise ValueError("%s would be %d bytes" % (path.stem, len(data)))
            (OUTPUT_DIR / (path.stem + ".out")).write_bytes(data)
        except Exception:  # noqa: BLE001 - log it and go on
            traceback.print_exc(file=sys.stderr)
            print("%s: no forecast written" % (path.name,), file=sys.stderr)


if __name__ == "__main__":
    main()
