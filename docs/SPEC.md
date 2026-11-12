# GF2 unit-source QC spec (pre-aggregation)

Goal: decide, per subduction zone, which unit-source HySEA runs are **ACCEPT / REVIEW / RERUN / EXCLUDE**
and which POIs go on a **global POI exclusion list**, before linear aggregation into EQ scenarios.

Design for speed: **each pkl is read exactly once** (stage 1, parallel, cached to a small npz of per-POI
stats). Stages 2–4 only use the cached npz files, so thresholds can be retuned and re-evaluated in seconds.
Plots are made only for non-ACCEPT runs.

Reference implementation to reuse: `~/gf2_qc_test/gf2_pkl_qc.py` (scan + cache), `gf2_qa_all.py` (zones),
`plot_run_diag.py`, `plot_truncation.py`.

---

## 0. Input facts (verified on southamerica run 501)

| item | value |
|---|---|
| root | `/leonardo_work/DTGEO_T1_2/ChEESE2_subductions_GF2/GF2_GEBCO_<zone>/<zone>/GF2/slip_taper_geo_NNN_hysea/` |
| files | `out_ts_new_th.pkl`, `out_ts_new_th_velocity.pkl`, `parfile.txt` |
| pkl type | dict |
| meta keys | `npois, has_velocity, all_valid_pois, start_time, end_time, ntimes, lon, lat, fault_1..fault_N` |
| POI keys | int `i` (1-based POI id) -> `numpy.ma.MaskedArray` float32; `crop_time_i` (s) = time of first stored sample |
| sizes | `len(all_valid_pois)` = POI list length (24458); `npois` = # stored series; `lon/lat` length = `npois`, lon in 0–360 |
| time | `start_time`=0, `end_time`≈259200 s, `ntimes`=17281, dt=15 s; series i spans `crop_time_i + k*dt`, k<len_i, and ends at `end_time` |
| source | `fault_k` string: `time, lon_barycenter, lat_barycenter, rake, slip` -> source centroid (slip-weighted mean) |
| POI file | `/leonardo_work/DTGEO_T1_2/VMg_runPTHA/POI_lon_lat_ID_depth_GEBCO.txt` (line i = POI i: lon lat ID depth) |
| python | `/leonardo_work/DTGEO_T1_2/VMg_runPTHA/.venv/bin/python` (numpy, pandas, matplotlib) |

Series are **cropped**: storage starts when the POI first exceeds the storage threshold, so POIs never
reached are absent (treat as 0 in aggregation) and the first stored value should be ~0 unless `crop_time`=0.

---

## 1. Thresholds (single config block)

```python
NT_EXPECTED = 17281          # 72 h / 15 s + 1
DT          = 15.0
FILL        = 1e6            # |eta| above = garbage
HIGH        = 10.0           # m, |eta| flag (store POI max/min)
JUMP2       = 5.0            # m, |eta[t+2]-eta[t]| (change within 2 steps = 30 s)
FLAT_MIN_H  = 1.0            # h, exactly-constant stretch after arrival
START_OFF   = 0.05           # m, |eta[0]| for series with crop_time > 0 (should start ~0)
END_OFF     = 0.10           # m, |mean(last 1 h)| for far-field POIs (dist > NEAR_KM)
GROW_RATIO  = 0.7            # rms(last 6 h) / rms(peak +-1 h) -> no decay
AMP_MIN     = 0.01           # m, ignore energy/offset tests below this peak
NEAR_KM     = 500            # near field (coseismic offset allowed)
FAR_KM      = 2000           # far field for "hot spot" logic
V_MAX       = 1.2 * 262 * 3.6   # km/h; sqrt(g*7000 m)=262 m/s (~943 km/h) +20 % margin -> ~1130 km/h
HOT_FAC     = 10.0           # POI pmax vs k-NN POI median (isolated explosion)
RES_FAC     = 1.0            # log10 residual above distance-decay envelope
SPK_WIN_MIN = 10             # min, spike-time clustering window
SPK_SPAN_KM = 3000           # spikes this far apart in same window = global instability
STUCK_FRAC  = 0.5            # POI constant in > 50 % of runs -> stuck
BADPOI_FRAC = 0.05           # POI flagged in > 5 % of runs (and >= 3 runs) -> pathological
REF_RATIO   = (0.2, 5.0)     # ref-POI pmax vs neighbour-source median
NB_CORR_MIN = 0.7            # corr(log pmax) with nearest source
```

---

## 2. Stage A — file integrity (per pkl, in the single read)

| id | check | flag | severity |
|---|---|---|---|
| A1 | pkl exists and `pickle.load` succeeds | `LOAD_ERROR` | RERUN |
| A2 | meta keys present (`npois, all_valid_pois, start_time, end_time, ntimes, lon, lat, fault_1`) | `MISSING_KEYS` | RERUN |
| A3 | `ntimes == 17281` and `end_time >= 259200 - DT`; also parfile sim time / save dt agree | `SHORT_NTIMES(n)` | RERUN |
| A4 | # int POI keys == `npois` == len(`lon`); every POI key has `crop_time_i` | `MISSING_TS(n)` | RERUN |
| A5 | any NaN / inf / masked value | `NAN(nPOI)` | RERUN |
| A6 | any \|eta\| > FILL | `FILL(nPOI)` | RERUN |
| A7 | any \|eta\| > HIGH (10 m) -> record list of (POI, max, min) | `HIGH(nPOI, max)` | REVIEW (RERUN if far field / with C flags) |
| A8 | record `npois`, `len(all_valid_pois)`, file size, source centroid, total slip | info | — |

---

## 3. Stage B — per-POI time-series stats (vectorised per series, stored in npz)

Per stored POI `i` (arrays of length `len(all_valid_pois)`, NaN where not stored):

| field | definition |
|---|---|
| `pmax, pmin` | max / min eta |
| `t_pk` | time of max \|eta\| (h) |
| `t_arr` | `crop_time_i` + first k with \|eta\| > 1e-3 (h) |
| `const` | `max == min` (whole series constant) |
| `jump2` | max \|eta[t+2] − eta[t]\|, and `t_jump2` |
| `flat_h` | longest run of `diff == 0` **after** `t_arr`, in hours |
| `off0` | \|eta[0]\| if `crop_time_i > 0` |
| `offend` | mean(eta over last 1 h) |
| `grow` | rms(last 6 h) / rms(t_pk ± 1 h) |
| `trunc` | `crop_time_i + (len−1)*dt < end_time − dt` or `len < 2` |

Per-POI flags (counted per run; POI ids stored for stage D):

| id | flag | rule |
|---|---|---|
| B1 | `CONST` | `const` (later split into stuck vs. run-specific) |
| B2 | `HIGH` | `pmax > HIGH` or `pmin < −HIGH` (keep value) |
| B3 | `SPIKE` | `jump2 > JUMP2` |
| B4 | `NO_DECAY` | `grow > GROW_RATIO` and `pmax > AMP_MIN` (energy growth / no decay) |
| B5 | `FLAT_AFTER_ARR` | `flat_h > FLAT_MIN_H` and not `const` |
| B6 | `START_OFFSET` | `off0 > START_OFF` (series jumps in at a non-zero level) |
| B7 | `END_OFFSET` | \|`offend`\| > END_OFF and dist > NEAR_KM (permanent offset in far field) |
| B8 | `TRUNC_TS` | `trunc` |

---

## 4. Stage C — per unit source spatial / propagation consistency

Inputs: centroid from `fault_*`, great-circle distance `d_i` (km) to each POI, `pmax`, `t_arr`, `t_jump2`.

| id | check | algorithm | flag |
|---|---|---|---|
| C1 | **causality** | apparent speed `d_i / t_arr_i > V_MAX` for `d_i > NEAR_KM` -> signal arrives before physically possible | `ACAUSAL(n)` |
| C2 | **distance-decay envelope** | bin log10(d) (e.g. 20 bins); envelope = p95 of log10(pmax) per bin, made monotone non-increasing for d > FAR_KM (cumulative max from far to near); residual = log10(pmax) − envelope(d); flag POIs with residual > RES_FAC and d > FAR_KM | `FAR_HOT(n)` |
| C3 | **isolated explosion** | k-NN (k=8) among stored POIs on unit-sphere xyz (`scipy.spatial.cKDTree`, else brute force on candidates only); flag `pmax > HOT_FAC × median(pmax_nn)` and `pmax > 0.1 m` | `ISOLATED_HOT(n)` |
| C4 | **spike-time clustering** | take POIs with B3 (or `jump2 > 2 m`); slide SPK_WIN_MIN window over `t_jump2`; if POIs in one window span > SPK_SPAN_KM -> synchronous global blow-up (run 502: ~21 h and ~35 h, Arctic + Banda + Tahiti) | `GLOBAL_INSTAB` |
| C5 | **footprint** | n POIs with pmax > 1e-3 and > 0.01 m vs. zone neighbours (stage D) | info |
| C6 | **map** (only non-ACCEPT runs + optional zone contact sheet) | scatter log10 pmax (≥1e-3) on lon/lat, source star, red rings = B3/C2/C3 POIs; 2nd panel `t_jump2` vs d | png |

Run-level spatial score: `frac_bad = (FAR_HOT ∪ ISOLATED_HOT ∪ ACAUSAL) / n_stored`.

---

## 5. Stage D — subduction-zone level (from cached npz only)

D1. **Stuck POIs**: POI `const` in > STUCK_FRAC of runs where stored (≥3 runs).
Known: 104, 202, 208, 220, 238, 266, 281, 368, 1350 (constant in all 26 zones).

D2. **Pathological POIs**: POI flagged by any of B2–B7, C1–C3 in > BADPOI_FRAC of runs and ≥ 3 runs,
**not** explained by one bad run. Also: GEBCO depth ≥ 0 or duplicate coordinates in the POI file.

D3. **Reference-POI neighbour similarity**: pick ~50 reference POIs automatically
(stored in > 90 % of zone runs, depth > 1000 m, never flagged, spread in azimuth from zone).
For each run, neighbour sources = 2–4 nearest unit-source centroids (not index ±1).
`ratio = pmax_ref(run) / median(pmax_ref(neighbours))`; flag if median ratio outside REF_RATIO
or > 20 % of ref POIs outside it -> `REF_MISMATCH`.

D4. **Full-field neighbour correlation**: Pearson r of log10(pmax) over POIs stored in both runs
with nearest-centroid source; `r < NB_CORR_MIN` -> `NB_DISSIMILAR`. Also `n_stored` vs. neighbours
(< 0.5× -> weak/truncated, > 2× -> energy leak).

D5. **Ensemble outliers**: robust z (median/MAD) of log10 amp_max, amp_p99, max jump2, n_stored > 4 -> `OUTLIER`.

D6. **Truncated runs**: neighbours' POIs with `t_pk` > cut-off and POIs stored only by neighbours
(`plot_truncation.py` logic) -> report count, always RERUN.

---

## 6. Decision rules

Stuck + pathological POIs are excluded **before** computing run flags.

| status | rule |
|---|---|
| **RERUN** | any A1–A6; A3/D6 truncation; `GLOBAL_INSTAB`; `frac_bad` > 0.1 %; any B2 HIGH with d > FAR_KM; B3 SPIKE at ≥ 5 POIs or any jump2 > 10 m |
| **EXCLUDE** | no non-stuck POI stored (EMPTY) **and** source verified on land / zero slip; else RERUN |
| **REVIEW** | 1–4 B3 POIs only in near field (d < NEAR_KM, t < 2 h); B2 in near field; B4/B5/B7 on < 1 % of POIs; D3/D4/D5 flags; FAR_HOT/ISOLATED below RERUN level |
| **ACCEPT** | none of the above |

Known calibration cases: southamerica 196 (1 step, fill) -> RERUN; southamerica 502 (55 spikes, synchronous
at 21/35 h globally) -> RERUN; sunda2 21/142/298/388 (stop at ~33 h) -> RERUN; makran2 123–151, sunda2 479–480
(EMPTY) -> check source; manus 36, philippine 11/16, sesulawesi 1 (one near-field jump ~2 m) -> REVIEW/ACCEPT.

---

## 7. Outputs

Per zone `OUT/<zone>/`:
- `per_run/run_NNNN.npz` — cached per-POI stats (stage A+B), rec json
- `qc_runs.csv` — one row per unit source: integrity info, npois, counts per flag, amp_max/p99, max jump2,
  frac_bad, nb_corr, ref_ratio, z-scores, `status`, `reasons`
- `qc_pois.csv` — per POI: n_runs_stored, n_const, n_flag per type, max_over_runs (+ run), stuck, pathological
- `qc_high.csv` — every (run, POI, pmax, pmin, d_km) with |eta| > 10 m
- `diag/run_NNN_{map,ts}.png` — only non-ACCEPT runs

Global `OUT/`:
- `poi_exclusion_global.csv` — POI, reason (stuck / pathological / dry / duplicate), n_zones
- `unit_source_status.csv` — zone, run, status, reasons (input for aggregation)
- `QC_SUMMARY.md` — counts per zone and status, flagged-run table

---

## 8. Script layout (suggested)

```
gf2_qc.py scan     --zones ... -j 8     # stage A+B(+C distances) -> npz cache (only pass touching pkl)
gf2_qc.py evaluate [--config qc.yaml]  # stages C+D+decision from npz -> csv/md (seconds)
gf2_qc.py plot     --status RERUN REVIEW
```

- Parallel over pkl with `multiprocessing.Pool` (6–8 on login node, more in a batch job); cache keyed on
  run id and invalidated by pkl mtime; write via tmp + `os.replace`.
- Per series: one `np.ma.filled(..., nan)` -> float32; all stats from `x`, `np.diff(x)`, `x[2:]-x[:-2]`;
  flat-run length via run-length of `diff == 0` on the post-arrival slice.
- Stage C needs only pmax, t_arr, t_jump2, coordinates -> computed in `evaluate`, not in the scan.
