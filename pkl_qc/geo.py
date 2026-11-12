"""Spherical geometry and a cached k-nearest-neighbour POI graph (numpy only)."""
import os

import numpy as np

R_EARTH = 6371.0


def unit_xyz(lon, lat):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gc_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km), broadcasting."""
    lo1, la1, lo2, la2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def max_span_km(lon, lat):
    """Largest pairwise distance of a (small) point set."""
    if len(lon) < 2:
        return 0.0
    return float(gc_km(lon[:, None], lat[:, None], lon[None], lat[None]).max())


def knn_graph(lon, lat, k, cache=None, chunk=512):
    """(n,k) indices and distances (km) of the k nearest other POIs; cached in `cache` npz."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["idx"].shape == (len(lon), k) and np.allclose(z["lon"], lon):
            return z["idx"], z["dist"]
    xyz = unit_xyz(lon, lat).astype(np.float32)
    n = len(lon)
    idx = np.empty((n, k), np.int32)
    dist = np.empty((n, k), np.float32)
    for s in range(0, n, chunk):
        dot = xyz[s:s + chunk] @ xyz.T
        dot[np.arange(dot.shape[0]), np.arange(s, s + dot.shape[0])] = -2.0   # drop self
        part = np.argpartition(-dot, k, axis=1)[:, :k]
        dd = np.take_along_axis(dot, part, 1)
        o = np.argsort(-dd, 1)
        idx[s:s + chunk] = np.take_along_axis(part, o, 1)
        dist[s:s + chunk] = R_EARTH * np.arccos(np.clip(np.take_along_axis(dd, o, 1), -1, 1))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, idx=idx, dist=dist, lon=lon)
    return idx, dist
