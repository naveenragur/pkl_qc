"""Diagnostic figures: per unit source (map, distance decay, spike timing, time series) and per zone."""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

from . import geo, scan  # noqa: E402
from .evaluate import FLAGS, distance_envelope  # noqa: E402

STATUS_COLOR = dict(ACCEPT="tab:green", REVIEW="tab:orange", RERUN="tab:red", EMPTY="tab:gray", EXCLUDE="k")
MARK = [("SPIKE", "red", "o"), ("FAR_HOT", "magenta", "s"), ("ISO_HOT", "darkviolet", "D"), ("ACAUSAL", "black", "^"),
        ("HIGH", "black", "x"), ("NO_DECAY", "saddlebrown", "v"), ("FLAT", "dimgray", "<"), ("END_OFF", "olive", ">")]
MCOL = {n: c for n, c, _ in MARK}
AMP_LEVELS = np.arange(-3.0, 1.51, 0.5)          # log10 max eta classes (half decades)


def _amp_norm(vmax):
    """Discrete colour classes for log10 max eta: easier to read than a continuous dark map."""
    lev = AMP_LEVELS[AMP_LEVELS <= max(vmax, -2.0) + 0.5]
    cols = plt.get_cmap("turbo")(np.linspace(0.12, 0.95, len(lev) - 1))      # skip turbo's near-black ends
    cmap = matplotlib.colors.ListedColormap(cols)
    return lev, cmap, matplotlib.colors.BoundaryNorm(lev, cmap.N)


def _len_label(v):
    """10**v metres as a short readable length: 3 mm, 1 cm, 30 cm, 3 m."""
    x = 10 ** v
    for unit, f in (("m", 1.0), ("cm", 0.01), ("mm", 0.001)):
        if x >= f * 0.999:
            return f"{x / f:.2g} {unit}"
    return f"{x:.1g} m"


def _series(d, i):
    nt, t0, t1 = int(d["ntimes"]), float(d["start_time"]), float(d["end_time"])
    dt = (t1 - t0) / max(nt - 1, 1)
    x = np.ma.filled(np.ma.asarray(d[i], float), np.nan)
    return (float(d.get(f"crop_time_{i}", 0.0)) + dt * np.arange(x.size)) / 3600.0, x


RC = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
      "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False}


def _g(row, k, default=np.nan):
    v = (row or {}).get(k, default)
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _flag_scatter(ax, x, y, n, c, mk, label=None, zorder=4):
    """Flag markers: large and open when few, small and solid when many (keeps dense maps readable)."""
    big = n < 150
    style = dict(s=70 if big else 26, marker=mk, linewidths=1.5 if big else 0.8, alpha=0.95 if big else 0.75)
    style.update(dict(c=c) if mk == "x" else dict(facecolors="none", edgecolors=c))
    return ax.scatter(x, y, label=label, zorder=zorder, **style)


def _info_text(rec, row, m, keep, cfg):
    f = lambda v, fmt: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)
    L = [("source", f"{f(rec.get('src_lon'), '.2f')}, {f(rec.get('src_lat'), '.2f')}  depth {f(rec.get('src_dep'), '.0f')} km"),
         ("", f"{rec.get('n_faults', 'n/a')} faults, slip max {f(rec.get('slip_max'), '.2f')} m"),
         ("time samples", f"{rec.get('ntimes', 'n/a')} / {cfg['nt_expected']}"),
         ("stored POIs", f"{rec.get('n_series', 'n/a')}  (kept {f(_g(row, 'n_kept'), '.0f')}, > 1 cm {f(_g(row, 'n_fp_1cm'), '.0f')})"),
         ("max / min eta", f"{f(_g(row, 'amp_max'), '.3g')} / {f(_g(row, 'eta_min'), '.3g')} m"),
         ("max 2-step jump", f"{f(_g(row, 'jump2_max'), '.3g')} m"),
         ("spike-time span", f"{f(_g(row, 'spike_span_km'), '.0f')} km  (limit {cfg['spk_span_km']:.0f})"),
         ("neighbours", str(_g(row, "nb_runs", "")) or "none"),
         ("corr log pmax", f"{f(_g(row, 'nb_corr'), '.2f')}  (limit {cfg['nb_corr_min']})"),
         ("ref POIs deviating", f"{f(100 * _g(row, 'ref_frac_out'), '.0f')} %  (limit {100 * cfg['ref_frac_out']:.0f} %)")]
    counts = [(n, int(((m & FLAGS[n]) > 0)[keep].sum())) for n in FLAGS if n != "CONST"]
    counts = [f"{n} {c}" for n, c in counts if c]
    L = [(k, v.replace("n/a m", "n/a").replace("n/a %", "n/a")) for k, v in L]
    return L, counts


def plot_run(zone, run, out, cfg, pois, row=None, with_ts=True):
    zdir = os.path.join(out, zone)
    rec, a = scan.load_cached(scan.cache_path(out, zone, run))
    if a is None:
        return None
    fl = np.load(os.path.join(zdir, "flags.npz"))
    r = int(np.flatnonzero(fl["runs"] == run)[0])
    m, excl = fl["M"][r], fl["excl"]
    plon, plat = pois["lon"], pois["lat"]
    st = a["stored"]
    keep = st & ~excl
    dist = geo.gc_km(rec["src_lon"], rec["src_lat"], plon, plat)
    pm = a["pmax"]
    status = str(_g(row, "status", ""))
    reasons = str(_g(row, "reasons", "") or "")
    with plt.rc_context(RC):
        f = _plot_run(zone, run, rec, a, m, excl, keep, dist, pm, plon, plat, status, reasons, out, cfg, pois, row,
                      with_ts, zdir)
    return f


def _plot_run(zone, run, rec, a, m, excl, keep, dist, pm, plon, plat, status, reasons, out, cfg, pois, row,
              with_ts, zdir):
    import textwrap
    nrow = 3 if with_ts else 2
    fig = plt.figure(figsize=(22, 17 if with_ts else 12), layout="constrained")
    gs = fig.add_gridspec(nrow, 4, height_ratios=[1.35, 1, 0.9][:nrow])
    head = fig.suptitle(f"{zone}  |  unit source {run}  |  {status or 'n/a'}", fontsize=16, fontweight="bold",
                        color=STATUS_COLOR.get(status, "k"))
    if reasons:
        fig.text(0.5, 0.995 - 0.022, "\n".join(textwrap.wrap(reasons.replace(";", ";  "), 190)),
                 ha="center", va="top", fontsize=10, color="0.25", transform=fig.transFigure)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.965))

    # ---- map of max eta + flagged POIs
    A = fig.add_subplot(gs[0, :3])
    A.scatter(plon, plat, s=0.3, c="0.88", zorder=1)
    s_ = keep & (pm > 1e-3)
    o = np.flatnonzero(s_)[np.argsort(pm[s_])]
    vmax = np.log10(max(np.nanmax(pm[o]), 1e-2)) if o.size else 0
    lev, cmap, norm = _amp_norm(vmax)
    sc = A.scatter(plon[o], plat[o], c=np.log10(pm[o]), s=11, cmap=cmap, norm=norm, linewidths=0, zorder=2)
    cb = fig.colorbar(sc, ax=A, orientation="horizontal", shrink=0.55, aspect=45, pad=0.02, ticks=lev,
                      label="max eta [m]  (stored POIs > 1 mm)")
    cb.ax.set_xticklabels([_len_label(v) for v in lev])
    ex = excl & a["stored"]
    handles = [A.scatter(plon[ex], plat[ex], s=60, marker="*", c="0.3", label=f"excluded POI ({ex.sum()})", zorder=3)]
    flags = sorted(((n, c, mk, keep & ((m & FLAGS[n]) > 0)) for n, c, mk in MARK), key=lambda t: -t[3].sum())
    for n, c, mk, fsel in flags:                       # most numerous first, rare flags on top
        if fsel.any():
            _flag_scatter(A, plon[fsel], plat[fsel], fsel.sum(), c, mk)
            # legend proxy with a fixed marker size (dense flags are drawn small on the map)
            handles.append(Line2D([], [], ls="", marker=mk, ms=9, color=c, mfc=c if mk == "x" else "none",
                                  mew=1.5, label=f"{n} ({fsel.sum()})"))
    handles.append(A.plot(rec["src_lon"], rec["src_lat"], "*", c="gold", ms=24, mec="k", mew=1.5, label="source",
                          zorder=6)[0])
    A.set(xlim=(-180, 180), ylim=(-80, 85), xlabel="lon", ylabel="lat", title="max eta per POI and flagged POIs")
    A.set_xticks(range(-180, 181, 60)); A.set_yticks(range(-60, 81, 30))
    A.grid(alpha=0.25, lw=0.5)

    # ---- info panel
    I = fig.add_subplot(gs[0, 3])
    I.axis("off")
    lines, counts = _info_text(rec, row, m, keep, cfg)
    y = 1.0
    I.text(0, y, "run summary", fontsize=12, fontweight="bold", va="top", transform=I.transAxes)
    y -= 0.075
    for k, v in lines:
        I.text(0.0, y, k, fontsize=9.5, color="0.35", va="top", transform=I.transAxes)
        I.text(0.42, y, v, fontsize=9.5, va="top", transform=I.transAxes)
        y -= 0.058
    I.text(0.0, y - 0.01, "flagged POIs", fontsize=9.5, color="0.35", va="top", transform=I.transAxes)
    rows_ = [", ".join(counts[i:i + 2]) for i in range(0, len(counts), 2)] or ["none"]
    I.text(0.42, y - 0.01, "\n".join(rows_), fontsize=9.5, va="top", linespacing=1.5, transform=I.transAxes,
           color="tab:red" if counts else "tab:green")
    I.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=8.5, markerscale=1.0,
             title="map legend", title_fontsize=9, alignment="left")

    # ---- distance decay
    B = fig.add_subplot(gs[1, 0])
    B.scatter(dist[keep], np.clip(pm[keep], 1e-5, None), s=2, c="0.6", zorder=1, label="POI")
    _, env, _, _ = distance_envelope(dist[keep & np.isfinite(pm)], pm[keep & np.isfinite(pm)], cfg)
    edges = np.linspace(np.log10(cfg["near_km"]), np.log10(20100.0), cfg["env_bins"] + 1)
    if np.isfinite(env).any():
        B.step(10 ** edges[:-1], 10 ** env, where="post", c="k", lw=1.5, label=f"p{cfg['env_q']:.0f} envelope")
        B.step(10 ** edges[:-1], 10 ** (env + cfg["res_fac"]), where="post", c="r", ls="--", lw=1.5,
               label="FAR_HOT limit")
    for n, c, mk, fsel in flags:
        if n in ("SPIKE", "FAR_HOT", "ISO_HOT", "ACAUSAL") and fsel.any():
            _flag_scatter(B, dist[fsel], np.clip(pm[fsel], 1e-5, None), fsel.sum(), c, mk)
    B.axvline(cfg["far_km"], c="0.5", ls=":", lw=1)
    B.set(xscale="log", yscale="log", xlabel="distance from source [km]", ylabel="max eta [m]",
          title="distance decay")
    B.legend(loc="lower left", frameon=False)
    B.grid(alpha=0.25, which="major", lw=0.5)

    # ---- causality
    D = fig.add_subplot(gs[1, 1])
    ta = a["t_arr_rel"]
    D.scatter(dist[keep], ta[keep], s=2, c="0.6", zorder=1)
    fa = keep & ((m & FLAGS["ACAUSAL"]) > 0)
    if fa.any():
        _flag_scatter(D, dist[fa], ta[fa], fa.sum(), MCOL["ACAUSAL"], "^", f"ACAUSAL ({fa.sum()})")
    D.plot([0, 20100], [0, 20100 / cfg["v_max_kmh"]], "k--", lw=1.2, label=f"{cfg['v_max_kmh']:.0f} km/h front")
    D.axvline(cfg["acausal_km"], c="0.5", ls=":", lw=1)
    D.set(xlim=(0, 20100), xlabel="distance from source [km]", ylabel="arrival time [h]",
          title="causality (points below the front = too early)")
    D.legend(loc="upper left", frameon=False)
    D.grid(alpha=0.25, lw=0.5)

    # ---- spike timing
    C = fig.add_subplot(gs[1, 2])
    cnd = keep & (np.nan_to_num(a["jump2"]) > cfg["plot_jump_min"])
    o = np.flatnonzero(cnd)[np.argsort(a["jump2"][cnd])]
    sc = C.scatter(a["t_jump2"][o], dist[o], c=np.log10(a["jump2"][o]), s=4, cmap="turbo",
                   vmin=np.log10(cfg["plot_jump_min"]), vmax=1.3)
    fig.colorbar(sc, ax=C, shrink=0.85, pad=0.01, aspect=30, label="log10 largest 30 s change |eta(t+30 s) - eta(t)| [m]")
    big = keep & (np.nan_to_num(a["jump2"]) > cfg["spk_cand"])
    if big.any():
        C.scatter(a["t_jump2"][big], dist[big], s=45, facecolors="none", edgecolors="r", lw=1.2,
                  label=f"jump > {cfg['spk_cand']:g} m ({big.sum()})")
    tt = np.linspace(0, 72, 100)
    C.plot(tt, tt * cfg["v_max_kmh"], "k--", lw=1.2, label="fastest front")
    C.set(xlim=(0, 72), ylim=(0, 20100), xlabel="time of largest 2-step jump [h]", ylabel="distance from source [km]",
          title="jump timing (vertical band far away = instability)")
    C.legend(loc="upper right", frameon=True, framealpha=0.85)
    C.grid(alpha=0.25, lw=0.5)
    if not cnd.any():
        _no_data(C, f"no POI with a 2-step jump > {cfg['plot_jump_min'] * 100:.0f} cm")

    # ---- decay / flat
    E = fig.add_subplot(gs[1, 3])
    g = a["grow"][keep]
    g = g[np.isfinite(g)]
    fh = a["flat_h"][keep]
    fh = fh[np.isfinite(fh)]
    nflat = int((fh > cfg["flat_min_h"]).sum())
    flat_txt = f"flat after arrival > {cfg['flat_min_h']:g} h: {nflat} POIs (longest {fh.max() if fh.size else 0:.2f} h)"
    if g.size:
        E.hist(np.log10(np.clip(g, 1e-4, None)), bins=50, color="tab:blue", alpha=0.75)
        E.axvline(np.log10(cfg["grow_ratio"]), c="r", ls="--", lw=1.2, label=f"NO_DECAY limit ({cfg['grow_ratio']:g})")
        E.plot([], [], " ", label=flat_txt)
        E.set_yscale("log")
        leg = E.legend(loc="upper left", frameon=True, framealpha=0.85, fontsize=8)
        if nflat:
            leg.get_texts()[-1].set_color("tab:red")
    else:
        _no_data(E, "no decay statistics (series too short)\n" + flat_txt)
    E.set_xlabel("log10 rms(last 6 h) / rms(peak +-1 h)")
    E.set_ylabel("# POIs")
    E.set_title("energy decay (right of red line = no decay)")
    E.grid(alpha=0.25, lw=0.5)

    if with_ts:
        _ts_row(fig, gs, zone, run, rec, a, m, keep, dist, out, cfg, pois, row, map_ax=A)
    dd = os.path.join(zdir, "diag")
    os.makedirs(dd, exist_ok=True)
    fpath = os.path.join(dd, f"run_{run:04d}.png")
    fig.savefig(fpath, dpi=90)
    plt.close(fig)
    return fpath


def _no_data(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="0.4",
            bbox=dict(fc="w", ec="0.7"))


def _neighbour(zone, row, out):
    """(run, cache arrays, run_dir) of the nearest neighbour unit source, or None."""
    nb = str((row or {}).get("nb_runs") or "").split()
    for n in nb:
        try:
            rec, a = scan.load_cached(scan.cache_path(out, zone, int(n)))
        except (OSError, ValueError):
            continue
        if a is not None:
            return int(n), a, rec["run_dir"]
    return None


def _ts_row(fig, gs, zone, run, rec, a, m, keep, dist, out, cfg, pois, row, map_ax=None):
    """Bottom row: flagged POIs, the POI deviating most from the nearest neighbour, one normal POI;
    nearest neighbour run overlaid at the same POI."""
    pm = a["pmax"]
    nb = _neighbour(zone, row, out)
    show = []
    bad = keep & ((m & ~np.uint16(FLAGS["CONST"])) > 0)
    if bad.any():
        score = np.where(bad, np.nan_to_num(a["jump2"]) + np.nan_to_num(np.abs(pm)), -np.inf)
        o = np.argsort(-score)
        show += [(int(j), "flagged") for j in o[:2] if np.isfinite(score[j])][:1 if nb else 2]
    if nb is not None:
        pn = nb[1]["pmax"]
        c = keep & nb[1]["stored"] & (np.nan_to_num(pois["depth"], nan=0) <= cfg["ref_max_depth"]) \
            & (np.nan_to_num(pm) > cfg["ref_min_amp"]) & (np.nan_to_num(pn) > cfg["ref_min_amp"])
        if c.sum() >= cfg["ref_min_n"]:
            lr = np.full(pm.size, np.nan)
            lr[c] = np.log10(pm[c] / pn[c])
            dev = np.abs(lr - np.nanmedian(lr))
            dev[[s_[0] for s_ in show]] = np.nan
            if np.isfinite(dev).any():
                j = int(np.nanargmax(dev))
                show.append((j, f"most deviating vs run {nb[0]}: {10 ** lr[j]:.2g}x (pair median {10 ** np.nanmedian(lr):.2g}x)"))
    pk = np.where(keep & (m == 0), pm, np.nan)
    if np.isfinite(pk).any():
        nrm = np.flatnonzero(pk > np.nanpercentile(pk, 95))
        nrm = nrm[~np.isin(nrm, [s_[0] for s_ in show])]
        if nrm.size:
            show.append((int(nrm[np.argmin(dist[nrm])]), "normal (largest near source)"))
    if not show:
        return
    with open(os.path.join(rec["run_dir"], cfg["pkl_name"]), "rb") as fh:
        d = pickle.load(fh)
    dn = None
    if nb is not None:
        try:
            with open(os.path.join(nb[2], cfg["pkl_name"]), "rb") as fh:
                dn = pickle.load(fh)
        except Exception:
            dn = None
    slots = gs[2, :].subgridspec(1, 3)
    for k, (j, why) in enumerate(show[:3]):
        X = fig.add_subplot(slots[0, k])
        t, x = _series(d, j + 1)
        few = x.size < 200
        X.plot(t, x, "k.-" if few else "k-", lw=0.8, ms=8 if x.size < 5 else 3,
               label=f"run {run}: max {np.nanmax(x):.3g} m" + (f" ({x.size} samples)" if few else ""), zorder=3)
        tn = xn = None
        if dn is not None and (j + 1) in dn:
            tn, xn = _series(dn, j + 1)
            X.plot(tn, xn, c="tab:orange", lw=1.0, alpha=0.9, label=f"nb run {nb[0]}: max {np.nanmax(xn):.3g} m", zorder=2)
        tags = [n for n in FLAGS if m[j] & FLAGS[n]]
        t0, t1 = _active_window(t, x, tn, xn)
        if m[j] & FLAGS["SPIKE"] and np.isfinite(a["t_jump2"][j]):
            X.axvline(a["t_jump2"][j], c="red", ls=":", lw=1.2, label=f"largest 30 s jump {a['jump2'][j]:.2g} m")
        X.set(xlim=(t0, t1), xlabel="time [h]" + ("" if (t0, t1) == (-0.5, 72) else f"  (zoom {t0:.1f}-{t1:.1f} h of 72 h)"),
              ylabel="eta [m]")
        X.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=14, steps=[1, 2, 2.5, 5, 10]))
        X.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        letter = "ABC"[k]
        if map_ax is not None:                      # mark the plotted POI on the map with the same letter
            map_ax.scatter(pois["lon"][j], pois["lat"][j], s=170, marker="o", facecolors="none",
                           edgecolors="k", linewidths=2.2, zorder=7)
            off = {"A": (-40, 30), "B": (40, 30), "C": (40, -35)}[letter]   # separate directions for close POIs
            map_ax.annotate(letter, (pois["lon"][j], pois["lat"][j]), xytext=off, textcoords="offset points",
                            fontsize=15, fontweight="bold", color="k", zorder=8, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", lw=1.2, alpha=0.95),
                            arrowprops=dict(arrowstyle="-", color="k", lw=1.3, shrinkA=0, shrinkB=7))
        X.text(-0.02, 1.13, letter, transform=X.transAxes, fontsize=20, fontweight="bold", ha="right", va="bottom",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="k", lw=1.2))
        X.set_title(f"POI {j + 1}  {pois['name'][j]}\n{dist[j]:.0f} km from source, depth {pois['depth'][j]:.0f} m",
                    fontsize=10, loc="left")
        X.text(0.01, 0.97, (", ".join(tags) if tags else "no flag") + "\n" + why, transform=X.transAxes,
               va="top", ha="left", fontsize=8.5, color="tab:red" if tags else "0.3",
               bbox=dict(fc="w", ec="0.85", alpha=0.85))
        X.legend(loc="lower right", frameon=True, framealpha=0.85)
        X.grid(alpha=0.3, lw=0.5)
        X.grid(alpha=0.12, lw=0.4, which="minor")


def _active_window(t, x, tn=None, xn=None, frac=0.1, pre=1.0, post=3.0, min_h=4.0, max_h=24.0):
    """Time window [h] around the main signal of this run: where |eta| > frac of its max, padded.
    Longer than max_h -> window around the peak (pre 4 h, after 20 h). Neighbour only used if this run is empty."""
    for tt, xx in ((t, x), (tn, xn)):
        if tt is None or xx is None or xx.size < 5 or not np.isfinite(xx).any():
            continue
        ax_ = np.abs(np.nan_to_num(xx))
        act = np.flatnonzero(ax_ > frac * ax_.max())
        if not act.size:
            continue
        t0, t1 = tt[act[0]] - pre, tt[act[-1]] + post
        if t1 - t0 > max_h:
            tp = tt[int(ax_.argmax())]
            t0, t1 = tp - 4.0, tp + max_h - 4.0
        if t1 - t0 < min_h:
            c = 0.5 * (t0 + t1)
            t0, t1 = c - min_h / 2, c + min_h / 2
        return round(max(t0, -0.5), 1), round(min(t1, 72.0), 1)
    return -0.5, 72.0


def plot_zone(zone, out, pois=None):
    zdir = os.path.join(out, zone)
    q = pd.read_csv(os.path.join(zdir, "qc_runs.csv"))
    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
    col = q["status"].map(STATUS_COLOR).fillna("k")
    flagged = q[q.status == "RERUN"]
    for (i, k), c, lab, log in (((0, 0), "amp_max", "max eta over kept POIs [m]", True),
                                ((0, 1), "n_kept", "# stored kept POIs", False),
                                ((1, 0), "nb_corr", "corr(log pmax) with nearest source", False),
                                ((1, 1), "ref_med_ratio", "ref POIs: pmax / neighbour median", True)):
        A = fig.add_subplot(gs[i, k])
        if c in q:
            A.scatter(q["run"], q[c], c=col, s=10)
            for _, r in flagged.iterrows():
                if np.isfinite(r.get(c, np.nan)):
                    A.annotate(str(int(r.run)), (r.run, r[c]), fontsize=7, color=STATUS_COLOR[r.status])
            if log:
                A.set_yscale("log")
            n = int(np.isfinite(q[c].values.astype(float)).sum())
            A.set(xlabel="unit source", ylabel=lab if n == len(q) else f"{lab}\n(n = {n} of {len(q)} runs)")
            A.grid(alpha=0.3)
        if (i, k) == (0, 0):
            for s_, c_ in STATUS_COLOR.items():
                A.scatter([], [], c=c_, label=f"{s_} ({(q.status == s_).sum()})")
            A.legend(fontsize=8, loc="best")

    # source centroids coloured by status
    M = fig.add_subplot(gs[:, 2])
    lon = ((q["src_lon"].values + 180) % 360) - 180
    lat = q["src_lat"].values
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.any():
        # unwrap across the dateline so the zone is contiguous
        if lon[ok].max() - lon[ok].min() > 180:
            lon = np.where(lon < 0, lon + 360, lon)
        pad = max(2.0, 0.15 * max(np.ptp(lon[ok]), np.ptp(lat[ok])))
        x0, x1 = lon[ok].min() - pad, lon[ok].max() + pad
        y0, y1 = lat[ok].min() - pad, lat[ok].max() + pad
        if pois is not None:
            pl = pois["lon"] if x1 <= 180 else np.where(pois["lon"] < 0, pois["lon"] + 360, pois["lon"])
            inb = (pl > x0) & (pl < x1) & (pois["lat"] > y0) & (pois["lat"] < y1)
            M.scatter(pl[inb], pois["lat"][inb], s=1, c="0.75", label="POIs (coast)")
        for s_ in ["ACCEPT", "REVIEW", "EXCLUDE", "EMPTY", "RERUN"]:      # problems drawn on top
            w = ok & (q["status"].values == s_)
            if w.any():
                M.scatter(lon[w], lat[w], s=40 if s_ != "ACCEPT" else 18, c=STATUS_COLOR[s_],
                          edgecolors="k", linewidths=0.4, label=s_, zorder=3)
        for _, r in q[q.status != "ACCEPT"].iterrows():
            j = int(np.flatnonzero(q.run.values == r.run)[0])
            if ok[j]:
                M.annotate(str(int(r.run)), (lon[j], lat[j]), xytext=(3, 3), textcoords="offset points",
                           fontsize=7, color=STATUS_COLOR[r.status], zorder=4)
        M.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="lon", ylabel="lat", aspect="equal",
              title="unit-source centroids (slip-weighted) by status")
        M.legend(fontsize=8, loc="best")
        M.grid(alpha=0.3)
    fig.suptitle(f"{zone}: {len(q)} unit sources")
    fig.tight_layout()
    f = os.path.join(zdir, "qc_zone.png")
    fig.savefig(f, dpi=100)
    plt.close(fig)
    return f
