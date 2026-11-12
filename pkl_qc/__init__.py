"""QC of GF2 unit-source HySEA POI time series (out_ts_new_th.pkl) before hazard aggregation."""
import os

# Login nodes cap CPU time per process; multithreaded BLAS burns it on idle spinning.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
