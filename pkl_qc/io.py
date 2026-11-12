"""Discovery of zones/runs, parfile + POI file parsing, source geometry."""
import glob
import os
import re

import numpy as np

from .geo import gc_km, unit_xyz


def find_zones(root, zone_glob, names=None, gf2_subdir="*/GF2"):
    """-> [(zone, gf2_dir)], zone = folder name without 'GF2_' (e.g. GEBCO_southamerica)."""
    out = []
    for zdir in sorted(glob.glob(os.path.join(root, zone_glob))):
        zone = os.path.basename(zdir.rstrip("/"))
        zone = zone[4:] if zone.startswith("GF2_") else zone
        if names and zone not in names and zone.split("_", 1)[-1] not in names:
            continue
        gf2 = sorted(glob.glob(os.path.join(zdir, gf2_subdir))) if gf2_subdir else [zdir]
        out.append((zone, gf2[0] if gf2 else None))
    return out


def run_id(path):
    m = re.search(r"_(\d+)_hysea", path)
    return int(m.group(1)) if m else -1


def find_runs(gf2_dir, run_glob):
    """-> sorted [(run, run_dir)], including dirs without a pkl."""
    dirs = glob.glob(os.path.join(gf2_dir, run_glob)) if gf2_dir else []
    return sorted((run_id(d), d) for d in dirs if os.path.isdir(d))


def read_parfile(run_dir):
    """-> dict(sim_time, ts_dt, faults=(n,4) [lon, lat, depth_km, slip])."""
    out = dict(sim_time=None, ts_dt=None, faults=None)
    try:
        lines = open(os.path.join(run_dir, "parfile.txt"), errors="ignore").read().splitlines()
    except OSError:
        return out
    nf, k0 = None, None
    for k, line in enumerate(lines):
        if "# Simulation time" in line:
            out["sim_time"] = float(line.split()[0])
        elif "# Saving time of time series" in line:
            out["ts_dt"] = float(line.split()[0])
        elif "# Number of faults" in line and nf is None:
            nf, k0 = int(line.split()[0]), k + 1
    if nf:
        f = []
        for line in lines[k0:k0 + nf]:
            t = line.split()
            try:
                v = [float(x) for x in t]
            except ValueError:
                break
            if len(v) == 12:      # triangular Okada: 0 lon lat dep x3 rake slip
                f.append([np.mean(v[1:10:3]), np.mean(v[2:10:3]), np.mean(v[3:10:3]), v[11]])
            elif len(v) >= 10:    # standard Okada: time lon lat depth length width strike dip rake slip
                f.append([v[1], v[2], v[3], v[-1]])
        out["faults"] = np.array(f) if f else None
    return out


def faults_from_pkl(d):
    """Fallback: 'fault_k' strings 'time: .., lon_barycenter: .., lat_barycenter: .., rake: .., slip: ..'."""
    f = []
    for k in d:
        if isinstance(k, str) and k.startswith("fault_"):
            kv = dict(re.findall(r"(\w+):\s*([-\d.eE+]+)", str(d[k])))
            try:
                f.append([float(kv["lon_barycenter"]), float(kv["lat_barycenter"]), np.nan, float(kv["slip"])])
            except KeyError:
                pass
    return np.array(f) if f else None


def source_summary(faults):
    """Slip-weighted centroid lon/lat/depth, radius (km) of the fault set, total/max slip."""
    if faults is None or len(faults) == 0:
        return dict(src_lon=np.nan, src_lat=np.nan, src_dep=np.nan, src_rad_km=np.nan,
                    slip_sum=np.nan, slip_max=np.nan, n_faults=0)
    lon, lat, dep, slip = faults.T
    w = np.clip(slip, 0, None)
    w = w if w.sum() > 0 else np.ones_like(w)
    v = (unit_xyz(lon, lat) * w[:, None]).sum(0)
    clon = np.degrees(np.arctan2(v[1], v[0]))
    clat = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    return dict(src_lon=float(clon), src_lat=float(clat),
                src_dep=float(np.nansum(dep * w) / w.sum()) if np.isfinite(dep).any() else np.nan,
                src_rad_km=float(gc_km(clon, clat, lon, lat).max()),
                slip_sum=float(slip.sum()), slip_max=float(slip.max()), n_faults=int(len(slip)))


def load_pois(path):
    """POI file line i (1-based) = POI i: lon lat ID depth. lon -> [-180, 180)."""
    lon, lat, name, dep = [], [], [], []
    for line in open(path):
        f = line.split()
        if not f:
            continue
        lon.append(float(f[0])); lat.append(float(f[1]))
        name.append(f[2] if len(f) > 2 else "")
        dep.append(float(f[3]) if len(f) > 3 else np.nan)
    lon = (np.array(lon) + 180.0) % 360.0 - 180.0
    return dict(lon=lon, lat=np.array(lat), name=np.array(name), depth=np.array(dep))
