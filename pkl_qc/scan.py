"""Stage A (file integrity) + B (per-POI time-series stats). The only stage that reads pkls.

Each pkl is read once; results go to <out>/<zone>/cache/run_NNNN.npz (invalidated when the pkl
or parfile is newer, or the cache version changes).
"""
import json
import os
import pickle
import time
import warnings

import numpy as np

from . import io
from .parallel import run_tasks

CACHE_VERSION = 2
META_KEYS = ("npois", "all_valid_pois", "start_time", "end_time", "ntimes", "lon", "lat")
FLOAT_FIELDS = ("pmax", "pmin", "crop", "t_pk", "t_arr", "t_arr_rel", "jump2", "t_jump2", "flat_h", "off0",
                "offend", "grow")
BOOL_FIELDS = ("stored", "const", "nan", "fill", "trunc")
SEC_H = 3600.0


def _longest_true(m):
    """Length of the longest run of True in a 1-D bool array."""
    if not m.any():
        return 0
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def series_stats(x, crop, dt, end_time, cfg):
    """Stats of one POI series x (float, NaN where masked) starting at time `crop` (s)."""
    s = dict(nan=bool(np.isnan(x).any()), fill=bool((np.abs(np.nan_to_num(x)) > cfg["fill"]).any()))
    n = x.size
    s["trunc"] = n < 2 or crop + (n - 1) * dt < end_time - 2 * dt
    ok = np.isfinite(x) & (np.abs(np.nan_to_num(x)) <= cfg["fill"])
    if not ok.any():
        return s
    if not ok.all():                      # keep positions; stats on valid samples only
        s.update(pmax=float(x[ok].max()), pmin=float(x[ok].min()), const=False)
        return s
    ax = np.abs(x)
    kpk = int(ax.argmax())
    s.update(pmax=float(x.max()), pmin=float(x.min()), t_pk=(crop + kpk * dt) / SEC_H)
    s["const"] = bool(n > 1 and s["pmax"] == s["pmin"])
    above = np.flatnonzero(ax > cfg["arr_thr"])
    ka = int(above[0]) if above.size else None
    if ka is not None:
        s["t_arr"] = (crop + ka * dt) / SEC_H
        kr = int(np.argmax(ax > max(cfg["arr_thr"], cfg["arr_rel"] * ax[kpk])))
        s["t_arr_rel"] = (crop + kr * dt) / SEC_H
    if n > 2:
        d2 = np.abs(x[2:] - x[:-2])
        k2 = int(d2.argmax())
        s["jump2"], s["t_jump2"] = float(d2[k2]), (crop + (k2 + 1) * dt) / SEC_H
    if ka is not None and n - ka > 2 and not s["const"]:
        s["flat_h"] = _longest_true(np.diff(x[ka:]) == 0) * dt / SEC_H
    s["off0"] = float(abs(x[0])) if crop > dt / 2 else 0.0
    n1h = int(round(SEC_H / dt))
    if n >= n1h:
        s["offend"] = float(x[-n1h:].mean())
    if ax[kpk] > cfg["amp_min"] and end_time - (crop + kpk * dt) > cfg["grow_min_lead_h"] * SEC_H:
        pk = x[max(kpk - n1h, 0):kpk + n1h]
        tail = x[-6 * n1h:]
        s["grow"] = float(np.sqrt(np.mean(tail ** 2)) / max(np.sqrt(np.mean(pk ** 2)), 1e-12))
    return s


def scan_run(run_dir, cfg, plon, plat):
    """-> (rec, arrays|None). rec holds integrity info and source geometry."""
    n_poi = len(plon)
    pkl = os.path.join(run_dir, cfg["pkl_name"])
    par = io.read_parfile(run_dir)
    rec = dict(run=io.run_id(run_dir), run_dir=run_dir, error="", missing_keys="",
               size_MB=os.path.getsize(pkl) / 1e6 if os.path.exists(pkl) else 0.0,
               par_sim_time=par["sim_time"], par_ts_dt=par["ts_dt"],
               nt_parfile=int(par["sim_time"] // par["ts_dt"]) + 1 if par["sim_time"] and par["ts_dt"] else None)
    faults = par["faults"]
    if not os.path.exists(pkl):
        rec["error"] = "missing pkl"
        rec.update(io.source_summary(faults))
        return rec, None
    try:
        with open(pkl, "rb") as fh:
            d = pickle.load(fh)
    except Exception as e:
        rec["error"] = repr(e)[:200]
        rec.update(io.source_summary(faults))
        return rec, None
    if faults is None:
        faults = io.faults_from_pkl(d)
    rec.update(io.source_summary(faults))
    miss = [k for k in META_KEYS if k not in d]
    rec["missing_keys"] = ";".join(miss)
    if miss and any(k in miss for k in ("ntimes", "end_time", "start_time")):
        rec["error"] = f"missing keys {miss}"
        return rec, None

    nt, t0, t1 = int(d["ntimes"]), float(d["start_time"]), float(d["end_time"])
    dt = (t1 - t0) / (nt - 1) if nt > 1 and t1 > t0 else (par["ts_dt"] or 15.0)
    ids = sorted(int(k) for k in d if isinstance(k, (int, np.integer)))
    n_valid = len(d["all_valid_pois"]) if "all_valid_pois" in d else None
    rec.update(ntimes=nt, start_time=t0, end_time=t1, dt=dt, npois=int(d.get("npois", -1)),
               n_series=len(ids), n_all_valid=n_valid,
               n_lonlat=len(d["lon"]) if "lon" in d else None,
               n_crop_missing=sum(f"crop_time_{i}" not in d for i in ids),
               n_out_of_range=sum(not 1 <= i <= n_poi for i in ids))
    # pkl lon/lat (one per stored series, id order) must match the POI file
    rec["lonlat_maxdiff_deg"] = None
    good_ids = [i for i in ids if 1 <= i <= n_poi]
    if "lon" in d and "lat" in d and len(d["lon"]) == len(ids) and good_ids == ids and ids:
        j = np.array(ids) - 1
        dlon = (np.asarray(d["lon"], float) - plon[j] + 180.0) % 360.0 - 180.0
        dlat = np.asarray(d["lat"], float) - plat[j]
        rec["lonlat_maxdiff_deg"] = float(np.nanmax(np.abs(np.concatenate([dlon, dlat]))))

    a = {k: np.full(n_poi, np.nan, np.float32) for k in FLOAT_FIELDS}
    a.update({k: np.zeros(n_poi, bool) for k in BOOL_FIELDS})
    for i in ids:
        if not 1 <= i <= n_poi:
            continue
        j = i - 1
        x = np.ma.filled(np.ma.asarray(d[i], dtype=np.float64), np.nan)
        crop = float(d.get(f"crop_time_{i}", 0.0))
        a["stored"][j], a["crop"][j] = True, crop
        for k, v in series_stats(x, crop, dt, t1, cfg).items():
            a[k][j] = v
    return rec, a


def cache_path(out, zone, run):
    return os.path.join(out, zone, "cache", f"run_{run:04d}.npz")


def load_cached(path):
    z = np.load(path, allow_pickle=False)
    rec = json.loads(str(z["rec"]))
    arr = None if "stored" not in z.files else {k: z[k] for k in z.files if k != "rec"}
    return rec, arr


def cached_scan(run_dir, cfg, plon, plat, out, zone, force=False):
    c = cache_path(out, zone, io.run_id(run_dir))
    pkl = os.path.join(run_dir, cfg["pkl_name"])
    newest = max([os.path.getmtime(p) for p in (pkl, os.path.join(run_dir, "parfile.txt")) if os.path.exists(p)],
                 default=0.0)
    if not force and os.path.exists(c) and os.path.getmtime(c) >= newest:
        try:
            rec, arr = load_cached(c)
            if rec.get("cache_version") == CACHE_VERSION and not str(rec.get("error", "")).startswith("scan failed"):
                return rec, arr
        except Exception:
            pass
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec, arr = scan_run(run_dir, cfg, plon, plat)
    rec.update(cache_version=CACHE_VERSION, scan_s=round(time.time() - t, 1))
    os.makedirs(os.path.dirname(c), exist_ok=True)
    tmp = c[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, rec=json.dumps(rec), **(arr or {}))
    os.replace(tmp, c)
    return rec, arr


_POI = {}


def _init(plon, plat):
    _POI["lon"], _POI["lat"] = plon, plat


def _worker(args):
    run_dir, cfg, out, zone, force = args
    try:
        rec, _ = cached_scan(run_dir, cfg, _POI["lon"], _POI["lat"], out, zone, force)
    except Exception as e:      # never stop the zone: record the failure, evaluate marks it RERUN
        rec = dict(run=io.run_id(run_dir), run_dir=run_dir, error=f"scan failed: {e!r}"[:200],
                   cache_version=CACHE_VERSION)
        rec.update(io.source_summary(io.read_parfile(run_dir)["faults"]))
        c = cache_path(out, zone, rec["run"])
        os.makedirs(os.path.dirname(c), exist_ok=True)
        np.savez_compressed(c, rec=json.dumps(rec))
    return rec


def scan_zone(zone, gf2_dir, cfg, pois, out, nworkers=6, force=False, runs=None, log=print):
    rl = io.find_runs(gf2_dir, cfg["run_glob"])
    if runs:
        rl = [(r, d) for r, d in rl if r in set(runs)]
    if not rl:
        log(f"[{zone}] no run dirs under {gf2_dir}")
        return []
    t0 = time.time()
    log(f"[{zone}] scanning {len(rl)} runs with {nworkers} workers")
    jobs = [(d, cfg, out, zone, force) for _, d in rl]
    recs = []
    # one pkl per worker process: each gets a fresh CPU-time budget on login nodes
    def died(job, exc):
        return dict(run=io.run_id(job[0]), run_dir=job[0], error=f"scan failed: worker died ({exc!r})"[:200])
    for k, rec in enumerate(run_tasks(_worker, jobs, nworkers, _init, (pois["lon"], pois["lat"]), died), 1):
        recs.append(rec)
        if k % 25 == 0 or k == len(jobs):
            log(f"[{zone}]   {k}/{len(jobs)}  {time.time() - t0:.0f}s")
    return sorted(recs, key=lambda r: r["run"])
