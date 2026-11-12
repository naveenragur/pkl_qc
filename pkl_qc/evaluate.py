"""Stages C (per-run spatial consistency), D (zone level) and the run decision. Cache-only, no pkl reads."""
import os
import warnings

import numpy as np
import pandas as pd

from . import geo, scan
from .geo import R_EARTH

warnings.filterwarnings("ignore", category=RuntimeWarning)

# per-POI flag bits
FLAGS = dict(CONST=1, HIGH=2, SPIKE=4, NO_DECAY=8, FLAT=16, START_OFF=32, END_OFF=64, TRUNC_TS=128,
             NANFILL=256, ACAUSAL=512, FAR_HOT=1024, ISO_HOT=2048)
B_SOFT = ("NO_DECAY", "FLAT", "START_OFF", "END_OFF")
SPATIAL = ("ACAUSAL", "FAR_HOT", "ISO_HOT")
POI_BAD = ("HIGH", "SPIKE") + B_SOFT + SPATIAL      # used for pathological-POI detection
STATUS_ORDER = ["RERUN", "EMPTY", "REVIEW", "EXCLUDE", "ACCEPT"]


def bits(*names):
    return np.uint16(sum(FLAGS[n] for n in names))


# ------------------------------------------------------------------ stage B+C per run
def distance_envelope(d, amp, cfg):
    """Monotone non-increasing p-quantile envelope of log10(amp) vs log10(distance)."""
    edges = np.linspace(np.log10(cfg["near_km"]), np.log10(20100.0), cfg["env_bins"] + 1)
    ld, la = np.log10(np.clip(d, 1, None)), np.log10(np.clip(amp, 1e-6, None))
    env = np.full(cfg["env_bins"], np.nan)
    b = np.clip(np.digitize(ld, edges) - 1, 0, cfg["env_bins"] - 1)
    inside = ld >= edges[0]
    for k in range(cfg["env_bins"]):
        s = inside & (b == k)
        if s.sum() >= cfg["env_min_n"]:
            env[k] = np.percentile(la[s], cfg["env_q"])
    if np.isnan(env).all():
        return edges, env, b, inside
    for k in range(1, len(env)):             # empty bins inherit the nearer bin
        if np.isnan(env[k]):
            env[k] = env[k - 1]
    first = np.flatnonzero(np.isfinite(env))[0]
    env[:first] = env[first]
    env = np.minimum.accumulate(env)          # far field may not exceed nearer field
    return edges, env, b, inside


def poi_flags(a, rec, dist, knn_idx, knn_dist, cfg):
    """uint16 flag bits per POI for one run (stuck/const POIs only get CONST)."""
    st = a["stored"]
    m = np.zeros(st.size, np.uint16)
    const = a["const"] & st
    m[const] |= FLAGS["CONST"]
    live = st & ~const
    pmax, pmin = a["pmax"], a["pmin"]
    amp = np.fmax(np.abs(pmax), np.abs(pmin))

    def put(name, cond):
        m[live & np.nan_to_num(cond, nan=0).astype(bool)] |= FLAGS[name]

    put("HIGH", (pmax > cfg["high"]) | (pmin < -cfg["high"]))
    put("SPIKE", a["jump2"] > cfg["jump2"])
    put("NO_DECAY", a["grow"] > cfg["grow_ratio"])
    put("FLAT", a["flat_h"] > cfg["flat_min_h"])
    put("START_OFF", a["off0"] > cfg["start_off"])
    put("END_OFF", (np.abs(a["offend"]) > cfg["end_off"]) & (dist > cfg["near_km"]) & (amp > cfg["amp_min"]))
    m[st & a["trunc"]] |= FLAGS["TRUNC_TS"]
    m[st & (a["nan"] | a["fill"])] |= FLAGS["NANFILL"]

    # C1 causality: apparent speed from the source edge
    deff = dist - np.nan_to_num(rec.get("src_rad_km") or 0.0)
    t = a["t_arr_rel"]
    put("ACAUSAL", (deff > cfg["acausal_km"]) & np.isfinite(t) & ((t <= 0) | (deff / np.where(t > 0, t, 1) > cfg["v_max_kmh"])))

    # C2 distance-decay envelope
    use = live & np.isfinite(pmax)
    if use.sum() > cfg["env_min_n"]:
        _, env, b, inside = distance_envelope(dist[use], pmax[use], cfg)
        if np.isfinite(env).any():
            res = np.full(st.size, -np.inf)
            iu = np.flatnonzero(use)
            res[iu[inside]] = np.log10(np.clip(pmax[iu[inside]], 1e-6, None)) - env[b[inside]]
            put("FAR_HOT", (dist > cfg["far_km"]) & (res > cfg["res_fac"]) & (pmax > cfg["far_hot_min"]))

    # C3 isolated hot POI: > hot_fac x every stored neighbour within knn_max_km (unstored = 0)
    cand = np.flatnonzero(live & (pmax > cfg["hot_min"]))
    if cand.size:
        nb, nd = knn_idx[cand], knn_dist[cand]
        within = nd <= cfg["knn_max_km"]
        nbv = np.where(st[nb] & within, np.nan_to_num(pmax[nb], nan=0.0), 0.0)
        nbv = np.where(within, nbv, -np.inf)
        ok = within.sum(1) >= cfg["knn_min_n"]
        hot = ok & (pmax[cand] > cfg["hot_fac"] * np.maximum(nbv.max(1), 1e-3))
        m[cand[hot]] |= FLAGS["ISO_HOT"]
    return m


def spike_cluster(a, keep, plon, plat, cfg):
    """C4: largest distance span (km) of POIs whose max 2-step jump falls in the same time window."""
    c = np.flatnonzero(keep & (np.nan_to_num(a["jump2"]) > cfg["spk_cand"]) & np.isfinite(a["t_jump2"]))
    if c.size < cfg["spk_min_n"]:
        return 0.0, np.nan, int(c.size)
    o = c[np.argsort(a["t_jump2"][c])]
    t = a["t_jump2"][o]
    w = cfg["spk_win_min"] / 60.0
    best, tbest = 0.0, np.nan
    j = 0
    for i in range(o.size):
        while t[i] - t[j] > w:
            j += 1
        if i - j + 1 >= cfg["spk_min_n"]:
            span = geo.max_span_km(plon[o[j:i + 1]], plat[o[j:i + 1]])
            if span > best:
                best, tbest = span, float(t[i])
    return best, tbest, int(c.size)


# ------------------------------------------------------------------ helpers
def robz(x):
    x = np.asarray(x, float)
    m = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - m))
    return (x - m) / (mad if mad > 0 else 1.0)


def integrity_flags(rec, n_poi, cfg):
    """Stage A flags (all mean RERUN)."""
    f = []
    if rec.get("error"):
        e = str(rec["error"])
        f.append("MISSING_PKL" if e == "missing pkl" else "SCAN_ERROR" if e.startswith("scan failed") else "LOAD_ERROR")
        return f
    if rec.get("missing_keys"):
        f.append(f"MISSING_KEYS({rec['missing_keys']})")
    nt, ne = rec.get("ntimes"), cfg["nt_expected"]
    if nt is not None and nt != ne:
        f.append(f"NTIMES({nt}/{ne})")
    if rec.get("nt_parfile") not in (None, ne):
        f.append(f"PARFILE_NT({rec['nt_parfile']})")
    if rec.get("n_series") != rec.get("npois"):
        f.append(f"MISSING_TS({rec.get('n_series')}/{rec.get('npois')})")
    if rec.get("n_crop_missing"):
        f.append(f"MISSING_CROP({rec['n_crop_missing']})")
    if rec.get("n_out_of_range"):
        f.append(f"BAD_POI_IDS({rec['n_out_of_range']})")
    if rec.get("n_all_valid") not in (None, n_poi):
        f.append(f"POI_LIST({rec['n_all_valid']}/{n_poi})")
    if (rec.get("lonlat_maxdiff_deg") or 0) > 1e-3:
        f.append(f"POI_COORDS({rec['lonlat_maxdiff_deg']:.3g}deg)")
    return f


def _src_dist(rec, plon, plat):
    lon, lat = rec.get("src_lon"), rec.get("src_lat")
    if lon is None or lat is None or not np.isfinite(lon) or not np.isfinite(lat):
        return np.full(plon.size, np.nan)
    return geo.gc_km(lon, lat, plon, plat)


def source_xyz(df):
    """3-D source centroid positions (km) for neighbour search; depth NaN -> 0."""
    dep = np.nan_to_num(df["src_dep"].values.astype(float))
    return geo.unit_xyz(df["src_lon"].values, df["src_lat"].values) * (R_EARTH - dep)[:, None]


# ------------------------------------------------------------------ zone evaluation
def evaluate_zone(zone, out, cfg, pois, knn, log=print):
    zdir = os.path.join(out, zone)
    cdir = os.path.join(zdir, "cache")
    files = sorted(f for f in os.listdir(cdir) if f.startswith("run_") and f.endswith(".npz")
                   and ".tmp" not in f) if os.path.isdir(cdir) else []
    if not files:
        log(f"[{zone}] no cache, run 'scan' first")
        return None
    plon, plat, pdep = pois["lon"], pois["lat"], pois["depth"]
    P, R = len(plon), len(files)
    kidx, kdist = knn

    # ---- pass 1: per-POI flags for every run (cache -> compact matrices)
    recs, M = [], np.zeros((R, P), np.uint16)
    PM, TP = np.full((R, P), np.nan, np.float32), np.full((R, P), np.nan, np.float32)
    ST, J2 = np.zeros((R, P), bool), np.full((R, P), np.nan, np.float32)
    TJ, PN = np.full((R, P), np.nan, np.float32), np.full((R, P), np.nan, np.float32)
    has = np.zeros(R, bool)
    for r, f in enumerate(files):
        rec, a = scan.load_cached(os.path.join(cdir, f))
        recs.append(rec)
        if a is None:
            continue
        has[r] = True
        dist = _src_dist(rec, plon, plat)
        M[r] = poi_flags(a, rec, dist, kidx, kdist, cfg)
        PM[r], PN[r], TP[r], ST[r], J2[r], TJ[r] = a["pmax"], a["pmin"], a["t_pk"], a["stored"], a["jump2"], a["t_jump2"]
    df = pd.DataFrame(recs).set_index("run", drop=False)
    runs = df["run"].values
    crit = [integrity_flags(rec, P, cfg) for rec in recs]
    nanfill_any = ((M & FLAGS["NANFILL"]) > 0).sum(1)
    full = has & np.array([not c for c in crit]) & (nanfill_any == 0)

    # ---- D1 stuck POIs
    nst = ST[full].sum(0)
    ncon = ((M[full] & FLAGS["CONST"]) > 0).sum(0)
    stuck = (ncon >= cfg["stuck_min_runs"]) & (ncon > cfg["stuck_frac"] * np.maximum(nst, 1))
    dry = np.nan_to_num(pdep, nan=-1.0) >= 0

    # C4 with stuck-only exclusion -> runs with global instability are ignored for POI statistics
    def c4(r, keep):
        a = dict(jump2=J2[r], t_jump2=TJ[r])
        return spike_cluster(a, ST[r] & keep, plon, plat, cfg)
    span0 = np.array([c4(r, ~stuck)[0] if has[r] else 0.0 for r in range(R)])
    clean = full & (span0 <= cfg["spk_span_km"])

    # ---- D2 pathological POIs (flagged in many clean runs)
    bad = bits(*POI_BAD)
    nflag = ((M[clean] & bad) > 0).sum(0)
    nst_c = ST[clean].sum(0)
    patho = ~stuck & (nflag >= cfg["badpoi_min_runs"]) & (nflag > cfg["badpoi_frac"] * np.maximum(nst_c, 1))
    excl = stuck | patho | dry
    keep = ~excl

    # ---- pass 2: run metrics on kept POIs
    rows = []
    for r in range(R):
        rec = recs[r]
        s = dict(zone=zone, run=int(runs[r]))
        s["integrity"] = ";".join(crit[r])
        if not has[r]:
            rows.append(s); continue
        k = ST[r] & keep
        dist = _src_dist(rec, plon, plat)
        mk = M[r] & np.where(k, np.uint16(0xFFFF), np.uint16(0))
        n_k = int(k.sum())
        pm = np.where(k, PM[r], np.nan)
        s.update(n_stored=int(ST[r].sum()), n_kept=n_k,
                 n_fp_1mm=int((pm > 1e-3).sum()), n_fp_1cm=int((pm > 1e-2).sum()),
                 amp_max=float(np.nanmax(pm)) if n_k else np.nan,
                 amp_p99=float(np.nanpercentile(pm, 99)) if n_k else np.nan,
                 eta_min=float(np.nanmin(np.where(k, PN[r], np.nan))) if n_k else np.nan,
                 jump2_max=float(np.nanmax(np.where(k, J2[r], np.nan))) if n_k else np.nan,
                 poi_amp_max=int(np.nanargmax(pm)) + 1 if n_k else -1)
        for name in FLAGS:
            s[f"n_{name}"] = int(((mk & FLAGS[name]) > 0).sum())
        s["n_NANFILL"] = int(((M[r] & FLAGS["NANFILL"]) > 0).sum())       # corruption counts on all POIs
        far = dist > cfg["far_km"]
        s["n_HIGH_far"] = int((((mk & FLAGS["HIGH"]) > 0) & far).sum())
        s["n_SPIKE_far"] = int((((mk & FLAGS["SPIKE"]) > 0) & far).sum())
        s["n_spatial"] = int(((mk & bits(*SPATIAL)) > 0).sum())
        s["frac_spatial"] = s["n_spatial"] / max(n_k, 1)
        s["frac_soft"] = int(((mk & bits(*B_SOFT)) > 0).sum()) / max(n_k, 1)
        span, tsp, ncand = c4(r, keep)
        s.update(spike_span_km=span, spike_span_t_h=tsp, n_spike_cand=ncand)
        rows.append(s)
    q = pd.DataFrame(rows).set_index("run", drop=False)
    for c in ("src_lon", "src_lat", "src_dep", "src_rad_km", "slip_sum", "slip_max", "n_faults", "size_MB",
              "ntimes", "end_time", "npois", "n_series", "error"):
        if c in df:
            q[c] = df[c].values

    # ---- D3/D4/D6 neighbour comparisons (sources nearest in 3-D)
    # runs already failing hard checks are neither neighbours nor part of the ensemble
    hard = np.array([decide(row, cfg)[0] == "RERUN" for row in q.to_dict("records")])
    valid = full & ~hard & (q["n_kept"].fillna(0).values > 0)
    xyz = source_xyz(q)
    vi = np.flatnonzero(valid & np.isfinite(xyz).all(1))
    nb_max = np.inf
    if vi.size > 1:
        dd = np.linalg.norm(xyz[vi][:, None] - xyz[vi][None], axis=2)
        np.fill_diagonal(dd, np.inf)
        nb_max = cfg["nb_max_fac"] * float(np.median(dd.min(1)))
    LP = np.where(ST & keep[None] & (PM > 1e-3), np.log10(np.clip(PM, 1e-6, None)), np.nan)
    ref_ok = keep & (np.nan_to_num(pdep, nan=0.0) <= cfg["ref_max_depth"])
    nkept = np.nan_to_num(q["n_kept"].values.astype(float))
    endt = np.nan_to_num(q["end_time"].values.astype(float)) if "end_time" in q else np.zeros(R)
    nb_runs = [""] * R
    nb_corr, nb_nsr, ref_med, ref_out, ref_n, tr_late, tr_nbo = (np.full(R, np.nan) for _ in range(7))
    ref_count, ref_dev_count = np.zeros(P, np.int32), np.zeros(P, np.int32)
    for r in range(R):
        if not has[r] or not np.isfinite(xyz[r]).all():
            continue
        others = vi[vi != r]
        if others.size == 0:
            continue
        d3 = np.linalg.norm(xyz[others] - xyz[r], axis=1)
        o = np.argsort(d3)[:cfg["nb_k"]]
        nb = others[o[d3[o] <= nb_max]]
        if nb.size == 0:
            continue
        nb_runs[r] = " ".join(str(runs[i]) for i in nb)
        if valid[r]:
            c = np.isfinite(LP[r]) & np.isfinite(LP[nb[0]])
            if c.sum() >= cfg["nb_min_common"]:
                nb_corr[r] = np.corrcoef(LP[r, c], LP[nb[0], c])[0, 1]
            nb_nsr[r] = nkept[r] / max(np.median(nkept[nb]), 1)
            dist = _src_dist(recs[r], plon, plat)
            meds, outs, used, devs = [], [], [], []
            for n in nb:
                refs = pair_refs(PM[r], PM[n], ST[r] & ST[n] & ref_ok, dist, cfg)
                if refs.size < cfg["ref_min_n"]:
                    continue
                # POI-based, scale-free: the overall level may differ between sources (depth, slip);
                # a problem shows as reference POIs that break the pair's common ratio
                lr = np.log10(PM[r, refs] / PM[n, refs])
                dev = np.abs(lr - np.median(lr)) > np.log10(cfg["ref_dev_fac"])
                meds.append(10 ** np.median(lr)); outs.append(dev.mean()); used.append(refs)
                devs.append(refs[dev])
            if meds:
                ref_med[r], ref_out[r] = np.median(meds), np.median(outs)
                ref_n[r] = np.median([u.size for u in used])
                ref_count[np.unique(np.concatenate(used))] += 1
                ref_dev_count[np.unique(np.concatenate(devs))] += 1
        if any(c.startswith("NTIMES") for c in crit[r]) and endt[r] > 0:
            cut = endt[r] / 3600.0
            late = (np.nan_to_num(TP[nb], nan=-1) > cut) & (np.nan_to_num(PM[nb]) > 1e-3) & keep[None]
            tr_late[r] = late.any(0).sum()
            tr_nbo[r] = (ST[nb].any(0) & ~ST[r] & keep).sum()
    q["nb_runs"], q["nb_corr"], q["nb_nstored_ratio"] = nb_runs, nb_corr, nb_nsr
    q["ref_med_ratio"], q["ref_frac_out"], q["ref_n"] = ref_med, ref_out, ref_n
    q["n_trunc_late"], q["n_trunc_nb_only"] = tr_late, tr_nbo

    # ---- D5 ensemble outliers
    for c in ("amp_max", "amp_p99", "jump2_max", "n_kept"):
        z = np.full(R, np.nan)
        z[vi] = robz(np.log10(np.clip(q[c].values[vi].astype(float), 1e-9, None)))
        q[f"z_{c}"] = z

    # ---- decision
    dec = [decide(row, cfg) for row in q.to_dict("records")]
    q["status"], q["reasons"] = [d[0] for d in dec], [d[1] for d in dec]

    # ---- POI table
    pt = pd.DataFrame(dict(poi=np.arange(1, P + 1), name=pois["name"], lon=plon, lat=plat, depth=pdep,
                           n_runs_stored=nst, n_runs_const=ncon))
    for name in POI_BAD:
        pt[f"n_{name}"] = ((M[clean] & FLAGS[name]) > 0).sum(0)
    pt["n_flag_any"] = nflag
    pt["frac_flag"] = nflag / np.maximum(nst_c, 1)
    PMf = np.where(ST[full], PM[full], np.nan)
    has_any = np.isfinite(PMf).any(0)
    pt["max_over_runs"] = np.nanmax(PMf, 0)
    pt["argmax_run"] = (np.where(has_any, runs[full][np.argmax(np.where(np.isfinite(PMf), PMf, -np.inf), 0)], -1)
                        if full.any() else -1)
    pt["median_over_runs"] = np.nanmedian(PMf, 0)
    pt["stuck"], pt["pathological"], pt["dry"] = stuck, patho, dry
    pt["n_used_as_ref"], pt["n_ref_dev"] = ref_count, ref_dev_count
    pt["excluded"] = excl

    # ---- |eta| > high table
    ri, pj = np.nonzero(ST & ((np.nan_to_num(PM) > cfg["high"]) | (np.nan_to_num(PN) < -cfg["high"])))
    hi = pd.DataFrame(dict(zone=zone, run=runs[ri], poi=pj + 1, name=pois["name"][pj], pmax=PM[ri, pj], pmin=PN[ri, pj],
                           d_km=[geo.gc_km(q["src_lon"].values[i], q["src_lat"].values[i], plon[j], plat[j])
                                 for i, j in zip(ri, pj)],
                           stuck=stuck[pj], excluded=excl[pj]))

    q.to_csv(os.path.join(zdir, "qc_runs.csv"), index=False)
    pt.to_csv(os.path.join(zdir, "qc_pois.csv"), index=False)
    hi.to_csv(os.path.join(zdir, "qc_high.csv"), index=False)
    np.savez_compressed(os.path.join(zdir, "flags.npz"), runs=runs, M=M, excl=excl, stuck=stuck, patho=patho)
    cnt = q["status"].value_counts().to_dict()
    log(f"[{zone}] {R} runs: " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in STATUS_ORDER)
        + f" | stuck POIs {int(stuck.sum())}, pathological {int(patho.sum())}, "
        f"ref POIs per pair (median) {np.nanmedian(ref_n) if np.isfinite(ref_n).any() else 0:.0f}")
    return q, pt, hi


def pair_refs(pm_r, pm_n, common, dist, cfg):
    """Reference POIs for one run/neighbour pair: POIs stored in both, top ref_n_pair by neighbour pmax,
    taken evenly from ref_dist_bins distance-quantile bins so near and far field both count."""
    c = np.flatnonzero(common & np.isfinite(dist) & (np.nan_to_num(pm_n) > cfg["ref_min_amp"]))
    if c.size <= cfg["ref_n_pair"]:
        return c
    nb = cfg["ref_dist_bins"]
    edges = np.quantile(dist[c], np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, dist[c], side="right") - 1, 0, nb - 1)
    per = int(np.ceil(cfg["ref_n_pair"] / nb))
    out = []
    for k in range(nb):
        ck = c[b == k]
        out.append(ck[np.argsort(-pm_n[ck])[:per]])
    return np.concatenate(out)


def _az_components(lon0, lat0, lon, lat):
    lo0, la0, lo, la = map(np.radians, (lon0, lat0, lon, lat))
    dl = lo - lo0
    return np.sin(dl) * np.cos(la), np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dl)


def decide(s, cfg):
    """-> (status, reasons) for one row of the run table."""
    rr, rv = [], []
    if s.get("integrity"):
        rr.append(s["integrity"])
    n_k = s.get("n_kept")
    if n_k is None or not np.isfinite(n_k):
        return ("RERUN", ";".join(rr) or "NO_DATA")
    if s["n_NANFILL"]:
        rr.append(f"NANFILL({s['n_NANFILL']})")
    if s["n_TRUNC_TS"]:
        rr.append(f"TRUNC_TS({s['n_TRUNC_TS']})")
    if n_k == 0 and not rr:
        if (s.get("slip_sum") or 0) <= 0:
            return "EXCLUDE", "ZERO_SLIP"
        return "EMPTY", "NO_POI_STORED(check source on land / storage threshold)"
    if s["spike_span_km"] > cfg["spk_span_km"]:
        rr.append(f"GLOBAL_INSTAB({s['spike_span_km']:.0f}km@{s['spike_span_t_h']:.1f}h)")
    if s["n_spatial"] >= cfg["rerun_min_bad"] and s["frac_spatial"] >= cfg["rerun_frac_bad"]:
        rr.append(f"SPATIAL({s['n_ACAUSAL']}/{s['n_FAR_HOT']}/{s['n_ISO_HOT']})")
    elif s["n_spatial"]:
        rv.append(f"SPATIAL({s['n_ACAUSAL']}/{s['n_FAR_HOT']}/{s['n_ISO_HOT']})")
    if s["n_HIGH_far"]:
        rr.append(f"HIGH_FAR({s['n_HIGH_far']})")
    elif s["n_HIGH"]:
        rv.append(f"HIGH_NEAR({s['n_HIGH']},max={s['amp_max']:.1f})")
    j2 = s["jump2_max"] if np.isfinite(s["jump2_max"]) else 0
    if s["n_SPIKE"] >= cfg["rerun_nspike"] or j2 > cfg["rerun_jump2"] or s["n_SPIKE_far"]:
        rr.append(f"SPIKE({s['n_SPIKE']},far={s['n_SPIKE_far']},max={j2:.1f})")
    elif s["n_SPIKE"]:
        rv.append(f"SPIKE_NEAR({s['n_SPIKE']},max={j2:.1f})")
    soft = [f"{n}({s[f'n_{n}']})" for n in B_SOFT if s[f"n_{n}"]]
    if soft:
        (rr if s["frac_soft"] >= cfg["rerun_frac_poi"] else rv).append("+".join(soft))
    c = s.get("nb_corr")
    if c is not None and np.isfinite(c) and c < cfg["nb_corr_min"]:
        rv.append(f"NB_CORR({c:.2f})")
    fo = s.get("ref_frac_out")
    if fo is not None and np.isfinite(fo) and fo > cfg["ref_frac_out"]:
        rv.append(f"REF_PATTERN(dev={fo:.0%},scale={s.get('ref_med_ratio', np.nan):.2f})")
    z = [f"z_{c}={s[f'z_{c}']:.1f}" for c in ("amp_max", "amp_p99", "jump2_max", "n_kept")
         if np.isfinite(s.get(f"z_{c}", np.nan)) and s[f"z_{c}"] > cfg["zlim"]]
    if z:
        rv.append("OUTLIER(" + ",".join(z) + ")")
    if rr:
        return "RERUN", ";".join(rr + rv)
    if rv:
        return "REVIEW", ";".join(rv)
    return "ACCEPT", ""


# ------------------------------------------------------------------ across zones
def summarize(out, zones, cfg, pois, log=print):
    qs, ps = [], []
    for z in zones:
        f = os.path.join(out, z, "qc_runs.csv")
        if os.path.exists(f):
            qs.append(pd.read_csv(f))
            p = pd.read_csv(os.path.join(out, z, "qc_pois.csv"))
            p["zone"] = z
            ps.append(p)
    if not qs:
        log("nothing to summarise")
        return
    q = pd.concat(qs, ignore_index=True)
    cols = ["zone", "run", "status", "reasons", "n_kept", "amp_max", "jump2_max", "nb_corr", "ref_med_ratio",
            "src_lon", "src_lat", "src_dep", "slip_sum", "ntimes", "size_MB"]
    q[[c for c in cols if c in q]].to_csv(os.path.join(out, "unit_source_status.csv"), index=False)

    p = pd.concat(ps, ignore_index=True)
    st, pa = p[p.stuck], p[p.pathological]
    ex = pd.DataFrame(index=pd.Index(np.arange(1, len(pois["lon"]) + 1), name="poi"))
    ex["n_zones"] = p.groupby("poi").size()
    ex["n_zones_stuck"] = st.groupby("poi").size()
    ex["n_zones_patho"] = pa.groupby("poi").size()
    ex["zones_stuck"] = st.groupby("poi")["zone"].agg(" ".join)
    ex["zones_patho"] = pa.groupby("poi")["zone"].agg(" ".join)
    ex["max_over_runs"] = p.groupby("poi")["max_over_runs"].max()
    ex = ex.fillna({"n_zones": 0, "n_zones_stuck": 0, "n_zones_patho": 0, "zones_stuck": "", "zones_patho": ""})
    for c in ("n_zones", "n_zones_stuck", "n_zones_patho"):
        ex[c] = ex[c].astype(int)
    ix = ex.index.values - 1
    ex["name"], ex["lon"], ex["lat"], ex["depth"] = pois["name"][ix], pois["lon"][ix], pois["lat"][ix], pois["depth"][ix]
    ex["dry"] = np.nan_to_num(ex["depth"].values, nan=-1) >= 0
    key = np.round(pois["lon"], 5) + 1j * np.round(pois["lat"], 5)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ex["duplicate_coords"] = cnt[inv][ix] > 1
    r_st = np.where(ex.n_zones_stuck > 0, "stuck(" + ex.n_zones_stuck.astype(str) + " zones)", "")
    r_pa = np.where(ex.n_zones_patho >= cfg["global_bad_zones"], "pathological(" + ex.n_zones_patho.astype(str) + " zones)", "")
    r_dr = np.where(ex.dry, "dry(depth>=0)", "")
    reason = [";".join(x for x in t if x) for t in zip(r_st, r_pa, r_dr)]
    ex["reason"] = reason
    ex["exclude"] = ex["reason"] != ""
    ex = ex.reset_index()
    flagged = ex[ex.exclude | (ex.n_zones_patho > 0) | ex.duplicate_coords]
    flagged.to_csv(os.path.join(out, "poi_exclusion_global.csv"), index=False)

    tab = q.pivot_table(index="zone", columns="status", values="run", aggfunc="count", fill_value=0)
    tab = tab.reindex(columns=[c for c in STATUS_ORDER if c in tab.columns])
    tab["n_runs"] = tab.sum(1)
    ns = p.groupby("zone").agg(stuck=("stuck", "sum"), pathological=("pathological", "sum"))
    tab = tab.join(ns)
    nonok = q[q.status.isin(["RERUN", "EMPTY", "REVIEW", "EXCLUDE"])].sort_values(["status", "zone", "run"],
                                                                                 key=_status_key)
    L = ["# pkl_qc summary", "",
         f"zones: {len(qs)}, unit sources: {len(q)}  |  " +
         ", ".join(f"{s} {int((q.status == s).sum())}" for s in STATUS_ORDER), "",
         "## Status per zone", "", "```", tab.to_string(), "```", "",
         f"## Global POI exclusion ({int(ex.exclude.sum())} POIs)", "", "```",
         ex[ex.exclude][["poi", "name", "lon", "lat", "depth", "reason", "zones_patho"]].to_string(index=False),
         "```", "",
         "Zone-specific pathological POIs (excluded only in that zone): "
         f"{int(((ex.n_zones_patho > 0) & (ex.n_zones_patho < cfg['global_bad_zones'])).sum())}, "
         "see poi_exclusion_global.csv", "",
         f"## Non-ACCEPT unit sources ({len(nonok)})", "", "```",
         nonok[["zone", "run", "status", "reasons"]].to_string(index=False, max_colwidth=160) if len(nonok) else "none",
         "```", ""]
    open(os.path.join(out, "QC_SUMMARY.md"), "w").write("\n".join(L))
    log("\n".join(L[:12]))
    log(f"wrote {out}/QC_SUMMARY.md, unit_source_status.csv, poi_exclusion_global.csv")


def _status_key(col):
    return col.map({s: i for i, s in enumerate(STATUS_ORDER)}) if col.name == "status" else col
