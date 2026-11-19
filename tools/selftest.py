"""tools/selftest.py -- prove the deliverable cannot be invalidated.

Two jobs:

  (default)   Adversarial cases, built in memory: no history, one read, nulls, 0.00% and
              100.00% rates, no players-tracked, a one-minute and a three-week horizon,
              hundreds of reads with real steps, a single trophy, string ids.  Each case
              goes through main.forecast_case and the output contract is checked row by
              row, plus the whole payload is built twice to prove the bytes repeat.

  --verify DIR  Run the real program the way the grader does (python3 -P -s main.py) twice
              over a directory of <id>.in cases into two fresh output directories, then
              check every output: parses, exactly the case's trophies, each id once, five
              finite non-decreasing numbers in 0..100, at most 65,536 bytes, and the two
              runs byte-identical (sha256).  It also reports the wall time of each run,
              which must stay far below the grader's 8 seconds.

Usage:
    python3 tools/selftest.py
    python3 tools/selftest.py --verify <input dir>
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as entry  # noqa: E402
import model  # noqa: E402

BASE = {
    "schema": "makefun/gta6-trophies-case/v1",
    "id": "t",
    "at": "2026-11-19T03:00:00Z",
    "horizon_minutes": 90,
    "target": "2026-11-19T04:30:00Z",
    "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
    "game": {"name": "Grand Theft Auto VI", "platform": "PS5", "release_date": "2026-11-19",
             "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
}
TROPHIES = [
    {"id": 1, "name": "Vice Got a Price!", "description": "", "tier": "platinum",
     "points": 300, "secret": False},
    {"id": 2, "name": "Just Like Paradise", "description": "", "tier": "bronze",
     "points": 15, "secret": True},
    {"id": 3, "name": "Paradise Lost", "description": "", "tier": "bronze",
     "points": 15, "secret": True},
    {"id": 4, "name": "Gold", "description": "", "tier": "gold", "points": 90,
     "secret": False},
]


def read(hours, rates, tracked=638, earned=943):
    """A history read at `hours` since 2026-11-19T00:00Z."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    stamp = (base + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"slot": stamp, "at": stamp, "players_tracked": tracked,
            "trophies_earned": earned, "rates": rates}


def cases():
    """(name, case) for every shape that has broken a model before."""
    out = []
    case = dict(BASE, id="one-read", trophies=TROPHIES,
                history=[read(0.0, [0.63, 100.0, 1.25, 1.1])])
    out.append(("one read", case))

    case = dict(BASE, id="no-history", trophies=TROPHIES, history=[])
    out.append(("no history", case))

    hist = [read(h, [0.63, 100.0, 1.25, 1.1]) for h in (0, 0.5, 1, 1.5, 2)]
    hist[2] = read(1, [None, 100.0, None, 1.1])
    case = dict(BASE, id="nulls", trophies=TROPHIES, history=hist)
    out.append(("nulls in the middle", case))

    hist = [read(0, [0.0, 100.0, 0.0, 0.0]), read(1, [0.0, 100.0, 0.01, 0.0]),
            read(2, [0.01, 100.0, 0.02, 0.01])]
    case = dict(BASE, id="zeros", trophies=TROPHIES, history=hist)
    out.append(("0.00 and 100.00 rates", case))

    hist = [{"slot": "2026-11-19T00:00:00Z", "at": "2026-11-19T00:00:00Z",
             "players_tracked": None, "trophies_earned": 0,
             "rates": [0.63, 100.0, 1.25, 1.1]},
            {"slot": "2026-11-19T01:00:00Z", "at": "2026-11-19T01:00:00Z",
             "players_tracked": 0, "trophies_earned": 0,
             "rates": [0.7, 100.0, 1.3, 1.2]}]
    case = dict(BASE, id="no-tracked", trophies=TROPHIES, history=hist)
    out.append(("no players tracked", case))

    case = dict(BASE, id="tiny-horizon", horizon_minutes=1, target="2026-11-19T03:01:00Z",
                trophies=TROPHIES,
                history=[read(h, [0.63, 100.0, 1.25, 1.1]) for h in (0, 1, 2, 3)])
    out.append(("one-minute horizon", case))

    case = dict(BASE, id="long-horizon", horizon_minutes=30240,
                target="2026-11-19T03:00:00Z", at="2026-11-07T15:00:00Z",
                trophies=TROPHIES,
                history=[read(h, [0.63, 100.0, 1.25, 1.1]) for h in (0, 1, 2, 3)])
    out.append(("three-week horizon", case))

    # a real staircase: 400 reads, a step every 40 reads, drifting down
    hist = []
    for k in range(400):
        h = k * (1.0 / 6.0)
        step = k // 40
        rates = [max(0.0, 0.63 * (0.75 ** step)), 100.0,
                 max(0.01, 1.25 * (0.78 ** step)), max(0.01, 1.1 * (0.8 ** step))]
        hist.append(read(h, rates, tracked=int(638 * (2.5 ** step)),
                         earned=int(943 * (2.0 ** step))))
    case = dict(BASE, id="staircase", at="2026-11-19T18:00:00Z",
                horizon_minutes=6510, target="2026-11-22T06:00:00Z",
                trophies=TROPHIES, history=hist)
    out.append(("400 reads, stepped", case))

    case = dict(BASE, id="one-trophy", trophies=TROPHIES[:1],
                history=[read(h, [0.63]) for h in (0, 1, 2)])
    out.append(("a single trophy", case))

    case = dict(BASE, id="string-ids",
                trophies=[dict(TROPHIES[0], id="a"), dict(TROPHIES[1], id="b")],
                history=[read(h, [0.63, 100.0]) for h in (0, 1, 2)])
    out.append(("string ids", case))

    case = dict(BASE, id="no-levels", trophies=TROPHIES,
                history=[read(h, [0.63, 100.0, 1.25, 1.1]) for h in (0, 1)])
    case.pop("levels")
    out.append(("levels key missing", case))
    return out


def check_payload(case, payload):
    """Everything the grader checks about one output file, raised as AssertionError."""
    assert isinstance(payload, dict) and list(payload) == ["trophies"], "extra keys"
    rows = payload["trophies"]
    wanted = [t["id"] for t in case["trophies"]]
    assert len(rows) == len(wanted), "row count %d != %d" % (len(rows), len(wanted))
    seen = []
    for row in rows:
        assert list(row) == ["id", "q"], "row keys %r" % (list(row),)
        seen.append(row["id"])
        q = row["q"]
        assert len(q) == 5, "five quantiles"
        for v in q:
            assert isinstance(v, (int, float)) and not isinstance(v, bool), "number"
            assert v == v and abs(v) != float("inf"), "finite"
            assert 0.0 <= v <= 100.0, "%r out of 0..100" % (v,)
        for k in range(1, 5):
            assert q[k] >= q[k - 1], "quantiles out of order: %r" % (q,)
        assert model.half_up(float(v)) == v or v in (0.0, 100.0, 97.0, 99.5), "4 decimals"
    assert sorted(seen) == sorted(wanted), "ids differ from the case's"
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    assert len(data) <= entry.MAX_BYTES, "payload %d bytes" % len(data)
    return data


def run_adversarial():
    failures = 0
    for name, case in cases():
        started = time.time()
        try:
            payload = entry.forecast_case(case, name)
            first = check_payload(case, payload)
            second = check_payload(case, entry.forecast_case(case, name))
            assert first == second, "not deterministic"
            assert payload["trophies"] or not case["trophies"], "no rows"
            took = time.time() - started
            print("  ok   %-24s %2d rows  %6d bytes  %.3fs"
                  % (name, len(payload["trophies"]), len(first), took / 2.0))
        except Exception as exc:  # noqa: BLE001 - a case that fails is the point
            failures += 1
            print("  FAIL %-24s %s: %s" % (name, type(exc).__name__, exc))
    return failures


def run_verify(directory):
    """Run main.py twice over a directory and compare the outputs byte for byte."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.in"))
    if not paths:
        print("no *.in files in %s" % (directory,))
        return 1
    failures = 0
    tmp = Path(tempfile.mkdtemp(prefix="makefun-verify-"))
    digests = []
    for run in (1, 2):
        out = tmp / ("out%d" % run)
        out.mkdir()
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(directory), MAKEFUN_OUTPUT_DIR=str(out))
        started = time.time()
        proc = subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")],
                              env=env, capture_output=True, text=True)
        took = time.time() - started
        if proc.returncode != 0:
            failures += 1
            print("  FAIL run %d exited %d: %s" % (run, proc.returncode, proc.stderr[-400:]))
        print("  run %d: %.2fs  stderr %r" % (run, took, proc.stderr.strip()[:200]))
        if took > 8.0:
            failures += 1
            print("  FAIL run %d took %.2fs" % (run, took))
        digest = {}
        for path in sorted(out.glob("*.out")):
            data = path.read_bytes()
            digest[path.name] = hashlib.sha256(data).hexdigest()
            case = json.loads((directory / (path.stem + ".in")).read_text(encoding="utf-8"))
            try:
                payload = json.loads(data.decode("utf-8"))
                check_payload(case, payload)
                print("  ok   %-8s %6d bytes  sha %s" % (path.name, len(data),
                                                         digest[path.name][:16]))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("  FAIL %s: %s: %s" % (path.name, type(exc).__name__, exc))
        if len(digest) != len(paths):
            failures += 1
            print("  FAIL run %d wrote %d of %d outputs" % (run, len(digest), len(paths)))
        digests.append(digest)
    if digests[0] != digests[1]:
        failures += 1
        print("  FAIL the two runs differ")
    else:
        print("  ok   two runs byte-identical (%d files)" % (len(digests[0]),))
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default=None, metavar="DIR")
    args = ap.parse_args()
    if args.verify:
        failures = run_verify(args.verify)
    else:
        print("adversarial cases:")
        failures = run_adversarial()
    print("FAILURES: %d" % (failures,))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
