"""Default paths and thresholds. Override any key with a JSON file (--config)."""
import json
import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

DEFAULTS = dict(
    # ---- data layout
    root="/leonardo_work/DTGEO_T1_2/ChEESE2_subductions_GF2",
    zone_glob="GF2_GEBCO_*",              # zone dirs: <root>/<zone_glob>
    gf2_subdir="*/GF2",                   # run container inside a zone dir ("" = runs directly in the zone dir)
    run_glob="slip_taper_geo_*_hysea",    # run dirs (unit source id = number before "_hysea")
    pkl_name="out_ts_new_th.pkl",
    poi_file="POI_lon_lat_ID_depth_GEBCO.txt",   # relative paths: from the config file's folder (default: configs/)

    # ---- A: file integrity
    nt_expected=17281,                    # 72 h / 15 s + 1
    fill=1e6,                             # |eta| above = garbage value
    high=10.0,                            # m, |eta| flag

    # ---- B: per-POI time series
    arr_thr=1e-3,                         # m, arrival = first |eta| above
    arr_rel=0.1,                          # relative arrival: first |eta| > max(arr_thr, arr_rel*max|eta|)
    jump2=5.0,                            # m, |eta[t+2]-eta[t]| spike
    flat_min_h=1.0,                       # h, exactly constant stretch after arrival
    start_off=0.05,                       # m, |eta[0]| of a cropped series
    end_off=0.10,                         # m, |mean(last 1 h)| beyond near_km
    grow_ratio=1.5,                       # rms(last 6 h) / rms(peak +-1 h); clean zones < 1.4, run 502 1.67
    grow_min_lead_h=12.0,                 # only test decay if peak is this long before end
    amp_min=0.01,                         # m, ignore decay/offset tests below this peak

    # ---- C: spatial consistency per unit source
    near_km=500.0,
    far_km=2000.0,
    v_max_kmh=1130.0,                     # sqrt(g*7000 m)=943 km/h +20 %
    acausal_km=2000.0,                    # causality tested only beyond this (static deformation tail nearer)
    env_bins=24, env_q=95.0, env_min_n=20,
    res_fac=1.0,                          # log10 above distance-decay envelope
    far_hot_min=0.05,                     # m, ignore FAR_HOT below
    knn_k=8, knn_max_km=200.0, knn_min_n=3,
    hot_fac=10.0, hot_min=0.1,            # isolated hot POI: pmax > hot_fac*median(neigh) and > hot_min
    spk_cand=2.0,                         # m, jump2 candidates for spike-time clustering
    spk_win_min=10.0, spk_span_km=3000.0, spk_min_n=3,

    # ---- plots
    plot_jump_min=0.02,                   # m, smallest 2-step jump drawn in the spike-timing panel

    # ---- D: zone level
    stuck_frac=0.5, stuck_min_runs=3,
    badpoi_frac=0.05, badpoi_min_runs=3,
    global_bad_zones=2,                   # pathological in >= N zones -> global exclusion
    # reference POIs are chosen per run/neighbour pair: stored in both, deeper than 100 m, not excluded
    ref_n_pair=50, ref_dist_bins=5, ref_min_n=10, ref_min_amp=1e-3,
    ref_max_depth=-100.0,
    ref_dev_fac=5.0,                      # ref POI deviates if its ratio is > 5x off the pair's median ratio
    ref_frac_out=0.2,                     # REVIEW if > 20 % of a run's ref POIs deviate
    nb_k=4, nb_corr_min=0.7, nb_min_common=50,
    nb_max_fac=3.0,                       # neighbours within nb_max_fac x median nearest-source spacing
    zlim=4.0,                             # one-sided: only unusually large runs are flagged

    # ---- decision
    rerun_frac_bad=0.001, rerun_min_bad=5,
    rerun_nspike=5, rerun_jump2=10.0,
    rerun_frac_poi=0.01,                  # B4-B7 on >= this fraction of stored POIs -> RERUN
)


def load_config(path=None, **override):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as fh:
            user = json.load(fh)
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}")
        base = os.path.dirname(os.path.abspath(path))
        if user.get("poi_file"):
            user["poi_file"] = os.path.join(base, os.path.expanduser(user["poi_file"]))
        cfg.update(user)
    cfg.update({k: v for k, v in override.items() if v is not None})
    cfg["poi_file"] = os.path.join(CONFIG_DIR, os.path.expanduser(cfg["poi_file"]))   # no-op if absolute
    return cfg
