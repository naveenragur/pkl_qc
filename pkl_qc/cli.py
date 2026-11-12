"""Command line: python -m pkl_qc {scan,evaluate,plot,all} [options]"""
import argparse
import os
import sys
import time
import traceback

import pandas as pd

from . import config, evaluate, geo, io, plots, scan
from .parallel import run_tasks

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _eval_zone(args):
    z, out, cfg, pois, knn = args
    msgs = []
    try:
        evaluate.evaluate_zone(z, out, cfg, pois, knn, msgs.append)
    except Exception:
        msgs.append(f"[{z}] evaluate FAILED\n{traceback.format_exc()}")
    return "\n".join(msgs)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pkl_qc", description=__doc__)
    ap.add_argument("cmd", choices=["scan", "evaluate", "plot", "all"],
                    help="scan: read pkls -> cache | evaluate: checks + decisions from cache | "
                         "plot: diag figures | all: scan+evaluate+plot")
    ap.add_argument("--zones", nargs="*", help="zone names, e.g. southamerica or GEBCO_southamerica (default: all)")
    ap.add_argument("--runs", type=int, nargs="*", help="restrict scan/plot to these unit-source ids")
    ap.add_argument("--root", help="override cfg root")
    ap.add_argument("--zone-glob", help="override cfg zone_glob, e.g. 'GF2_SRTM_*'")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--config", help="JSON file overriding keys of pkl_qc/config.py")
    ap.add_argument("-j", type=int, default=6, help="parallel pkl readers (login node: <= 8)")
    ap.add_argument("--force", action="store_true", help="rescan even if cache is up to date")
    ap.add_argument("--status", nargs="*", default=["RERUN", "REVIEW"], help="statuses to plot")
    ap.add_argument("--max-plots", type=int, default=30, help="max run figures per zone")
    ap.add_argument("--no-ts", action="store_true", help="plots without time series (no pkl read)")
    a = ap.parse_args(argv)

    cfg = config.load_config(a.config, root=a.root, zone_glob=a.zone_glob)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "pkl_qc.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"=== pkl_qc {a.cmd} {time.strftime('%Y-%m-%d %H:%M:%S')} zones={a.zones or 'all'} out={a.out}")
    pois = io.load_pois(cfg["poi_file"])
    zones = io.find_zones(cfg["root"], cfg["zone_glob"], a.zones, cfg["gf2_subdir"])
    if not zones:
        sys.exit(f"no zones match {a.zones} under {cfg['root']}/{cfg['zone_glob']}")

    if a.cmd in ("scan", "all"):
        for z, gf2 in zones:
            scan.scan_zone(z, gf2, cfg, pois, a.out, a.j, a.force, a.runs, log)

    if a.cmd in ("evaluate", "all"):
        knn = geo.knn_graph(pois["lon"], pois["lat"], cfg["knn_k"],
                            cache=os.path.join(a.out, f"_poi_knn_k{cfg['knn_k']}.npz"))
        jobs = [(z, a.out, cfg, pois, knn) for z, _ in zones]
        for msg in run_tasks(_eval_zone, jobs, a.j, on_error=lambda j, e: f"[{j[0]}] evaluate FAILED: worker died ({e!r})"):
            log(msg)
        done = sorted(d for d in os.listdir(a.out) if os.path.exists(os.path.join(a.out, d, "qc_runs.csv")))
        evaluate.summarize(a.out, done, cfg, pois, log)

    if a.cmd in ("plot", "all"):
        jobs = []
        for z, _ in zones:
            f = os.path.join(a.out, z, "qc_runs.csv")
            if not os.path.exists(f):
                continue
            log(f"[{z}] zone plot {plots.plot_zone(z, a.out, pois)}")
            q = pd.read_csv(f).set_index("run", drop=False)
            sel = q[q.status.isin(a.status)] if not a.runs else q[q.run.isin(a.runs)]
            jobs += [(z, row.to_dict(), a.out, cfg, pois, not a.no_ts) for _, row in sel.head(a.max_plots).iterrows()]
        if jobs:
            # one figure per fresh process (pkl read for time series; CPU limit on login nodes)
            for msg in run_tasks(_plot_run, jobs, a.j,
                                 on_error=lambda j, e: f"[{j[0]}] run {int(j[1]['run'])} plot FAILED: worker died ({e!r})"):
                log(msg)


def _plot_run(args):
    z, row, out, cfg, pois, with_ts = args
    try:
        return f"[{z}] run {int(row['run'])} -> {plots.plot_run(z, int(row['run']), out, cfg, pois, row, with_ts)}"
    except Exception as e:
        return f"[{z}] run {int(row['run'])} plot failed: {e!r}"


if __name__ == "__main__":
    main()
