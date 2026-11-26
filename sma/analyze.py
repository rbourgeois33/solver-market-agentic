"""Matrix analysis: properties that drive the choice of Krylov method, AMG flavour and parameters.

Everything is computed on the CPU from the binary CSR. Cheap checks always run;
eigenvalue estimates (Lanczos) only with eigs=True.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .csr_io import read_csr


def _stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {}
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "median": float(np.median(x))}


def symmetry(A: sp.csr_matrix) -> dict:
    At = A.T.tocsr()
    P = A.copy(); P.data[:] = 1.0
    Pt = At.copy(); Pt.data[:] = 1.0
    pattern_diff = (P - Pt)
    pattern_diff.eliminate_zeros()
    D = (A - At)
    nA = spla.norm(A)
    rel = float(spla.norm(D) / nA) if nA > 0 else 0.0
    return {
        "structurally_symmetric": pattern_diff.nnz == 0,
        "pattern_mismatch_entries": int(pattern_diff.nnz),
        "rel_asymmetry_fro": rel,          # ||A - A^T||_F / ||A||_F
        "numerically_symmetric": rel < 1e-12,
    }


def diagonal(A: sp.csr_matrix) -> dict:
    d = A.diagonal()
    absrow = np.asarray(abs(A).sum(axis=1)).ravel()
    off = absrow - np.abs(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        dom = np.where(off > 0, np.abs(d) / off, np.inf)
    finite = dom[np.isfinite(dom)]
    nz_d = np.abs(d[d != 0])
    return {
        "zero_diagonal": int((d == 0).sum()),
        "positive_diagonal": int((d > 0).sum()),
        "negative_diagonal": int((d < 0).sum()),
        "abs_diag": _stats(np.abs(d)),
        "diag_ratio_max_min": float(nz_d.max() / nz_d.min()) if nz_d.size else None,
        "diag_dominance": {
            "rows_dominant(>=1)": int((dom >= 1).sum()),
            "rows_strictly_dominant(>1)": int((dom > 1).sum()),
            "ratio_stats_finite": _stats(finite),
        },
        "rows_with_only_diagonal": int((off == 0).sum()),  # e.g. Dirichlet rows
    }


def offdiag_signs(A: sp.csr_matrix) -> dict:
    C = A.tocoo()
    off = C.row != C.col
    v = C.data[off]
    return {"offdiag_positive": int((v > 0).sum()), "offdiag_negative": int((v < 0).sum()),
            "offdiag_zero_stored": int((v == 0).sum())}


def row_sums(A: sp.csr_matrix) -> dict:
    rs = A @ np.ones(A.shape[1])
    absrow = np.asarray(abs(A).sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(absrow > 0, np.abs(rs) / absrow, 0.0)
    return {
        "rel_row_sum": _stats(rel),        # |sum_j a_ij| / sum_j |a_ij|
        "rows_zero_sum(rel<1e-12)": int((rel < 1e-12).sum()),
        "note": "many zero row sums => constant vector close to the null space (Laplacian-like); "
                "all rows => singular (pure Neumann)",
    }


def structure(A: sp.csr_matrix) -> dict:
    nnz_row = np.diff(A.indptr)
    C = A.tocoo()
    band = np.abs(C.row.astype(np.int64) - C.col.astype(np.int64))
    return {
        "nrows": int(A.shape[0]), "ncols": int(A.shape[1]), "nnz": int(A.nnz),
        "nnz_per_row": _stats(nnz_row.astype(float)),
        "empty_rows": int((nnz_row == 0).sum()),
        "bandwidth": int(band.max()) if band.size else 0,
        "mean_abs_offset": float(band.mean()) if band.size else 0.0,
        "abs_value": _stats(np.abs(A.data)),
    }


def block_size(A: sp.csr_matrix, candidates=range(2, 9)) -> dict:
    """Detect a constant point-block structure (dof interleaving): for block size b, all
    rows in each group of b consecutive rows share the same block-column pattern."""
    n = A.shape[0]
    found = []
    rows = np.repeat(np.arange(n), np.diff(A.indptr))
    for b in candidates:
        if n % b:
            continue
        bc = A.indices // b
        # signature of the set of block columns of each row: (count of distinct, hash sum)
        key = rows.astype(np.int64) * (A.shape[1] // b + 1) + bc
        uniq = np.unique(key)
        urow = uniq // (A.shape[1] // b + 1)
        ubc = uniq % (A.shape[1] // b + 1)
        cnt = np.bincount(urow, minlength=n)
        h = np.bincount(urow, weights=(ubc * 2654435761 % 4294967291).astype(np.float64), minlength=n)
        cnt = cnt.reshape(-1, b); h = h.reshape(-1, b)
        same = np.all(cnt == cnt[:, :1], axis=1) & np.all(h == h[:, :1], axis=1)
        frac = float(same.mean())
        if frac > 0.99:
            found.append({"block_size": b, "fraction_consistent_blocks": frac})
    return {"candidates_consistent": found,
            "note": "a consistent b>1 suggests a system of PDEs with interleaved dofs: "
                    "try block/nodal AMG (hypre NumFunctions/Nodal, MueLu 'number of equations', "
                    "PETSc MatSetBlockSize)"}


def eig_estimates(A: sp.csr_matrix, sym: bool) -> dict:
    """Lanczos/Arnoldi estimates on the Jacobi-scaled operator D^-1/2 A D^-1/2 (sym) or D^-1 A."""
    t0 = time.perf_counter()
    d = A.diagonal()
    if np.any(d == 0):
        return {"skipped": "zero diagonal"}
    out = {}
    try:
        if sym and np.all(d > 0):
            s = 1.0 / np.sqrt(d)
            S = sp.diags(s) @ A @ sp.diags(s)
            lmax = spla.eigsh(S, k=1, which="LA", tol=1e-3, maxiter=2000, return_eigenvectors=False)
            out["lambda_max_scaled"] = float(lmax[0])
            try:
                lmin = spla.eigsh(S, k=1, which="SA", tol=1e-2, maxiter=3000, return_eigenvectors=False)
                out["lambda_min_scaled"] = float(lmin[0])
                out["cond_estimate_scaled"] = float(lmax[0] / lmin[0]) if lmin[0] > 0 else None
                out["spd_estimate"] = bool(lmin[0] > 0)
            except spla.ArpackNoConvergence as e:
                out["lambda_min_scaled"] = "not converged (small eigenvalues clustered near 0: "\
                                           "typical of a large elliptic problem)"
                if len(e.eigenvalues):
                    out["lambda_min_scaled_partial"] = float(np.min(e.eigenvalues))
        else:
            S = sp.diags(1.0 / d) @ A
            lm = spla.eigs(S, k=1, which="LM", tol=1e-3, maxiter=2000, return_eigenvectors=False)
            out["spectral_radius_jacobi_scaled"] = float(abs(lm[0]))
    except spla.ArpackNoConvergence:
        out["error"] = "ARPACK did not converge"
    out["time_s"] = time.perf_counter() - t0
    return out


def hints(r: dict) -> list[str]:
    h = []
    sym = r["symmetry"]
    dg = r["diagonal"]
    n = r["structure"]["nrows"]
    if sym["numerically_symmetric"] and dg["negative_diagonal"] == 0 and dg["zero_diagonal"] == 0:
        h.append("symmetric with positive diagonal: likely SPD -> CG is the natural Krylov method "
                 "(needs a symmetric preconditioner: symmetric smoothers, same pre/post sweeps)")
    elif sym["numerically_symmetric"]:
        h.append("symmetric but diagonal not all positive: indefinite? -> MINRES-like or GMRES")
    else:
        h.append(f"nonsymmetric (rel asymmetry {sym['rel_asymmetry_fro']:.2e}) -> GMRES/BiCGStab; "
                 "AMG with nonsymmetric-aware coarsening/interpolation")
    if dg["zero_diagonal"]:
        h.append(f"{dg['zero_diagonal']} zero diagonal entries: Jacobi/GS smoothers break -> "
                 "saddle point? needs special treatment")
    off = r["offdiag_signs"]
    if off["offdiag_positive"] < 0.01 * max(1, off["offdiag_negative"]) and dg["negative_diagonal"] == 0:
        h.append("almost no positive off-diagonals: M-matrix-like (e.g. diffusion/pressure Poisson) "
                 "-> classical AMG (hypre BoomerAMG) and smoothed aggregation both well suited")
    zs = r["row_sums"]["rows_zero_sum(rel<1e-12)"]
    if zs > 0.5 * n:
        h.append(f"{zs}/{n} rows have zero row sum: constant vector is (near) null space; "
                 "default AMG near-null-space (constant) is appropriate")
        if zs == n:
            h.append("ALL rows sum to zero: matrix is singular (pure Neumann). The rhs must be "
                     "compatible; consider a null-space projection")
    if dg["rows_with_only_diagonal"]:
        h.append(f"{dg['rows_with_only_diagonal']} rows with only a diagonal entry (Dirichlet-like)")
    if r["block_size"]["candidates_consistent"]:
        bs = r["block_size"]["candidates_consistent"][0]["block_size"]
        h.append(f"consistent {bs}x{bs} block structure: consider nodal/block AMG")
    ratio = dg.get("diag_ratio_max_min")
    if ratio and ratio > 1e6:
        h.append(f"diagonal varies by {ratio:.1e}: strong coefficient jumps/anisotropy -> strength "
                 "threshold and smoother choice matter")
    return h


def analyze(csr_path: str, eigs: bool = False) -> dict:
    t0 = time.perf_counter()
    A = read_csr(csr_path, mmap=False)
    t_read = time.perf_counter() - t0
    r = {"file": str(Path(csr_path).resolve()), "read_time_s": t_read}
    r["structure"] = structure(A)
    r["symmetry"] = symmetry(A)
    r["diagonal"] = diagonal(A)
    r["offdiag_signs"] = offdiag_signs(A)
    r["row_sums"] = row_sums(A)
    r["block_size"] = block_size(A)
    if eigs:
        r["eigen_estimates"] = eig_estimates(A, r["symmetry"]["numerically_symmetric"])
    r["hints"] = hints(r)
    r["total_time_s"] = time.perf_counter() - t0
    return r


def summary(r: dict) -> str:
    s = r["structure"]
    lines = [
        f"n={s['nrows']} nnz={s['nnz']} nnz/row={s['nnz_per_row']['mean']:.2f} "
        f"(min {s['nnz_per_row']['min']:.0f}, max {s['nnz_per_row']['max']:.0f}) bandwidth={s['bandwidth']}",
        f"symmetric: structural={r['symmetry']['structurally_symmetric']} "
        f"numerical={r['symmetry']['numerically_symmetric']} "
        f"(rel asym {r['symmetry']['rel_asymmetry_fro']:.2e})",
        f"diagonal: +{r['diagonal']['positive_diagonal']} -{r['diagonal']['negative_diagonal']} "
        f"0:{r['diagonal']['zero_diagonal']}, dominant rows "
        f"{r['diagonal']['diag_dominance']['rows_dominant(>=1)']}",
        f"offdiag: +{r['offdiag_signs']['offdiag_positive']} -{r['offdiag_signs']['offdiag_negative']}",
    ]
    if "eigen_estimates" in r:
        lines.append(f"eigs: {json.dumps(r['eigen_estimates'])}")
    lines.append("hints:")
    lines += [f"  - {h}" for h in r["hints"]]
    return "\n".join(lines)
