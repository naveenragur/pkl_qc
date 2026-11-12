# pkl_qc

QC of GF2 unit-source HySEA POI time series (`out_ts_new_th.pkl`) before hazard aggregation.
This is a standalone project with no imports from older QC scripts. The design is in [docs/SPEC.md](docs/SPEC.md).

## Run

Everything runs interactively (login node or workstation); there is no batch script.

```bash
cd <pkl_qc folder>
# GEBCO zones -> results_gebco/ ; SRTM zones -> results_srtm/
./run_pkl_qc.sh all      --config configs/gebco.json --out results_gebco -j 8          # scan + evaluate + plot, all zones
./run_pkl_qc.sh all      --config configs/srtm.json  --out results_srtm  -j 8
./run_pkl_qc.sh all      --config configs/gebco.json --out results_gebco --zones southamerica sunda2   # selected zones
./run_pkl_qc.sh scan     --config configs/gebco.json --out results_gebco --zones southamerica --runs 196 502   # selected runs
./run_pkl_qc.sh evaluate --config configs/gebco.json --out results_gebco               # re-apply checks from cache (fast)
./run_pkl_qc.sh plot     --config configs/gebco.json --out results_gebco --zones sunda2 --status RERUN REVIEW
```

- Use the same `--config` and `--out` for all steps of one dataset. Without `--config`, the defaults in `pkl_qc/config.py` are used (Leonardo GEBCO) and output goes to `results/`.
- Long runs: `nohup ./run_pkl_qc.sh all --config configs/gebco.json --out results_gebco -j 8 > gebco.out 2>&1 &`, or use `tmux`.
- An interrupted scan resumes where it stopped: runs already in the cache are not re-read.
- Python: `PKL_QC_PYTHON=/path/to/python ./run_pkl_qc.sh ...`. By default the wrapper uses the Leonardo venv `/leonardo_work/DTGEO_T1_2/forGEM_pygthm_naveen/venv/bin/python3.11` if it exists, else `python3`. It needs numpy, pandas and matplotlib ([requirements.txt](requirements.txt)).
- Spack instead of a venv: use `./run_pkl_qc_spack.sh` with the same arguments. It loads Spack (`module load spack`, or `$SPACK_ROOT/share/spack/setup-env.sh`), activates the environment given by `PKL_QC_SPACK_ENV=/path/to/env` (or `SPACK_ENV_DEFAULT` in the script), and checks that its `python3` has numpy, pandas and matplotlib.
- Temporary/cache files stay inside the project (`.cache/`) and the output folder; nothing is written to `/tmp`.

Login-node limits (Leonardo: 600 s CPU time and 1 GB memory per process):
- Each pkl, zone evaluation and figure runs in its own fresh process, so the limits apply per task. If a worker is killed, the task is recorded as failed (`SCAN_ERROR` / `plot FAILED`) instead of hanging; re-run the command to retry it.
- The largest pkl (about 1.7 GB, 22k POIs) takes about 20 s. Keep `-j` at 8 or below on a login node.

## New machine or new dataset

1. Copy the whole `pkl_qc` folder (the `results*` and `.cache` folders are not needed).
2. Install Python ≥ 3.9 with `pip install -r requirements.txt`, or point `PKL_QC_PYTHON` to an existing environment.
3. Copy [configs/template_new_machine.json](configs/template_new_machine.json) to `configs/<name>.json` and set:

| key | meaning |
|---|---|
| `root` | folder containing the zone folders |
| `zone_glob` | pattern for zone folders inside `root`, e.g. `GF2_*`, `GF2_SRTM_*` |
| `gf2_subdir` | pattern from a zone folder to the folder holding the run folders (`*/GF2` on Leonardo: `GF2_GEBCO_tolo/tolo/GF2`); `""` if the run folders are directly in the zone folder |
| `run_glob` | run folder pattern; the unit-source id is the number before `_hysea` |
| `pkl_name` | pkl file name in each run folder |
| `poi_file` | POI list used by the simulations: one line per POI id (line i = POI i): `lon lat name depth`, depth negative in water. A relative path is read from the config file's folder; the GEBCO and SRTM files are bundled in `configs/`, so normally only the file name is needed. |
| `nt_expected` | expected number of time samples (simulation time / saving step + 1) |

   Any other key of [pkl_qc/config.py](pkl_qc/config.py) (thresholds) can be added to the same file.
4. Check discovery on one small zone first, then run everything:

```bash
./run_pkl_qc.sh all --config configs/<name>.json --out results_<name> --zones <small zone> -j 4
./run_pkl_qc.sh all --config configs/<name>.json --out results_<name> -j 8
```

Expected per run folder: `parfile.txt` (HySEA, for the fault geometry and time settings) and the pkl with keys `npois, all_valid_pois, start_time, end_time, ntimes, lon, lat`, one masked array per stored POI id, and `crop_time_<id>`. If the parfile is missing, the source centroid is taken from the pkl `fault_k` entries.

## Stages

| stage | reads | what |
|---|---|---|
| `scan` | pkl + parfile | A (integrity) and B (per-POI stats). Writes `results/<zone>/cache/run_NNNN.npz` (~0.3 MB/run). |
| `evaluate` | cache | B flags, C (spatial checks per run), D (zone level), decision, global POI list |
| `plot` | cache (+ pkl for time series) | zone overview and per-run diagnostic figures |

Thresholds are set in [pkl_qc/config.py](pkl_qc/config.py). To change them, add the keys to your config JSON and rerun `evaluate` (and `plot`); no rescan is needed.

## Checks

**A. File integrity** (any failure → RERUN):
- the pkl loads and has all required keys;
- `ntimes == 17281`, and the parfile gives the same value;
- the number of stored series equals `npois`, and every series has a `crop_time`;
- POI ids and lon/lat match the POI file;
- no NaN, masked or fill (|eta| > 1e6) values.

**B. Per-POI time series** (each POI gets a flag bitmask):

| flag | rule |
|---|---|
| CONST | the whole series is constant |
| HIGH | max > 10 m or min < -10 m. Values go to `qc_high.csv`. |
| SPIKE | \|eta[t+2] - eta[t]\| > 5 m |
| NO_DECAY | rms over the last 6 h / rms around the peak (±1 h) > 1.5. Only tested when the peak is > 1 cm and at least 12 h before the end of the run. |
| FLAT | the series is exactly constant for > 1 h after the arrival |
| START_OFF | a cropped series starts with \|eta\| > 5 cm |
| END_OFF | \|mean of the last hour\| > 10 cm more than 500 km from the source |
| TRUNC_TS | the series ends before `end_time` |

**C. Spatial checks per unit source.** The source centroid is the slip-weighted mean of the parfile faults.

| flag | rule |
|---|---|
| ACAUSAL | beyond 2000 km, the wave (first time \|eta\| exceeds both 1 mm and 10 % of the peak) arrives faster than 1130 km/h |
| FAR_HOT | beyond 2000 km, log10 pmax is more than 1 above the distance-decay envelope (p95 per distance bin, forced to be non-increasing with distance) |
| ISO_HOT | pmax > 0.1 m and more than 10× every stored neighbour within 200 km (8 nearest POIs) |
| GLOBAL_INSTAB | POIs with jumps > 2 m whose jumps fall in the same 10-minute window are more than 3000 km apart (the run 502 pattern) |

**D. Zone level:**
- **stuck POIs:** constant in > 50 % of the runs where they are stored;
- **pathological POIs:** flagged in > 5 % of clean runs, and in at least 3 runs;
- **dry POIs:** GEBCO depth ≥ 0;
- **reference-POI neighbour similarity (POI-based):** for each run and each of its ≤ 4 nearest unit sources (3-D centroid distance, within 3× the zone's median source spacing; runs already at RERUN are never used as neighbours), the reference POIs are the POIs stored in both runs, not excluded and deeper than 100 m. Up to 50 are taken, the largest by neighbour pmax, spread evenly over 5 distance bins. For each reference POI, the ratio pmax(run)/pmax(neighbour) is compared with the pair's median ratio, so a different overall level from depth or slip is allowed. A POI deviates if its ratio is off by more than 5×. More than 20 % deviating reference POIs gives REF_PATTERN. `qc_pois.csv` counts how often each POI was used as a reference (`n_used_as_ref`) and how often it deviated (`n_ref_dev`).
- **full-field neighbour correlation:** correlation of log pmax with the nearest source, < 0.7 gives NB_CORR. The stored-POI ratio to the neighbours (`nb_nstored_ratio`) is reported as information only.
- **robust z-scores** (one-sided, high side only) of amp_max, amp_p99, jump2_max and n_kept above 4 give OUTLIER;
- **truncated runs:** count the POIs whose peak, in the neighbouring runs, comes after the cut-off.

**Decision rules:**

| status | when |
|---|---|
| RERUN | any A failure; GLOBAL_INSTAB; ≥ 5 spatial-flag POIs (and ≥ 0.1 %); HIGH or SPIKE beyond 2000 km; ≥ 5 SPIKE POIs or a jump > 10 m; B soft flags on ≥ 1 % of POIs |
| EMPTY | no non-excluded POI stored: check whether the source is on land |
| EXCLUDE | zero slip |
| REVIEW | near-field HIGH/SPIKE, a few spatial or soft flags, NB_CORR, REF_PATTERN, OUTLIER |
| ACCEPT | none of the above |

Stuck, pathological and dry POIs are removed before the run flags are counted.

## Outputs (in the `--out` folder)

| file | content |
|---|---|
| `QC_SUMMARY.md` | status counts per zone, global POI exclusion list, all non-ACCEPT runs with their reasons |
| `unit_source_status.csv` | zone, run, status and reasons: **input for aggregation** |
| `poi_exclusion_global.csv` | POIs with a reason (stuck / pathological in ≥ 2 zones / dry) and an `exclude` flag. It also lists zone-specific pathological POIs and duplicate coordinates. |
| `<zone>/qc_runs.csv` | all per-run metrics and flag counts |
| `<zone>/qc_pois.csv` | per-POI flag counts across runs, max/median over runs, stuck/pathological/ref |
| `<zone>/qc_high.csv` | every (run, POI) with \|eta\| > 10 m |
| `<zone>/qc_zone.png` | per-run metrics vs unit-source id (max eta, stored POIs, neighbour correlation, reference-POI ratio) and a map of unit-source centroids coloured by status |
| `<zone>/diag/run_NNNN.png` | per RERUN/REVIEW run. Header: status (coloured) and reasons. Row 1: global map of max eta with flagged POIs, and a run summary (source, integrity, amplitudes, neighbour checks, flag counts, map legend). Row 2: distance decay with envelope and FAR_HOT limit; causality (arrival time vs distance with the fastest physical front); timing of the largest 30 s jump vs distance (a vertical band far from the source means instability); energy-decay histogram with the NO_DECAY limit and the flat-series count. Row 3: time series of flagged POIs, the POI deviating most from the nearest neighbour run, and a normal near-source POI, each overlaid with the nearest neighbour run (orange) |
| `pkl_qc.log` | log of every command |

## Layout

```
pkl_qc/config.py    thresholds and paths
pkl_qc/io.py        zone/run discovery, parfile faults, POI file
pkl_qc/geo.py       great-circle distance, POI kNN graph
pkl_qc/scan.py      stages A+B (only module that reads pkls), npz cache
pkl_qc/evaluate.py  stages C+D, decision, global summary
pkl_qc/plots.py     figures
pkl_qc/parallel.py  one process per task, fails instead of hanging if a worker dies
pkl_qc/cli.py       command line
configs/            gebco.json, srtm.json (Leonardo), template_new_machine.json,
                    POI_lon_lat_ID_depth_GEBCO.txt / _SRTM.txt (24,458 POIs, same coordinates, bathymetry-specific depths)
run_pkl_qc.sh       wrapper: picks the python, runs `python -m pkl_qc`
docs/SPEC.md        original QC specification
```
