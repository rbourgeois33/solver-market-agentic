"""Convert a linear system (MatrixMarket or PETSc binary) to the SMA binary format.

Output directory layout (a "system"):
  A.csr        matrix (sma.csr_io format)
  b.vec        right-hand side
  system.json  provenance: sources, sha256 of outputs, shape, nnz, conversion timings
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from .csr_io import read_csr, read_vec, write_csr, write_vec

PETSC_MAT_ID = 1211216
PETSC_VEC_ID = 1211214


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _petsc_int_width(path: Path, header_n: int, nz: int, nrows: int) -> int:
    """PETSc binary ints are 32-bit unless PETSc was built with 64-bit indices: infer from size. The file
    may continue after the matrix with other PETSc objects (some applications append the right-hand side
    Vec): then the matrix must be followed by a Vec class id."""
    size = path.stat().st_size
    for w in (4, 8):
        end = 4 * w + nrows * w + nz * w + nz * 8
        if size == end:
            return w
        if size > end:
            with open(path, "rb") as f:
                f.seek(end)
                nxt = np.fromfile(f, ">i4" if w == 4 else ">i8", 1)
            if len(nxt) and int(nxt[0]) == PETSC_VEC_ID:
                return w
    raise ValueError(f"{path}: size does not match a PETSc AIJ binary with 32 or 64-bit ints")


def read_petsc_mat(path: Path) -> sp.csr_matrix:
    with open(path, "rb") as f:
        hdr = np.fromfile(f, ">i4", 4)
        if hdr[0] == PETSC_MAT_ID:  # 32-bit PetscInt header (could still be 64-bit data: checked by size)
            m, n, nz = int(hdr[1]), int(hdr[2]), int(hdr[3])
        else:  # PETSc built with 64-bit indices writes the header as int64
            f.seek(0)
            hdr = np.fromfile(f, ">i8", 4)
            if hdr[0] != PETSC_MAT_ID:
                raise ValueError(f"{path}: not a PETSc Mat binary (classid {hdr[0]})")
            m, n, nz = int(hdr[1]), int(hdr[2]), int(hdr[3])
        w = _petsc_int_width(path, 4, nz, m)
        f.seek(4 * w)
        it = ">i4" if w == 4 else ">i8"
        rowlen = np.fromfile(f, it, m).astype(np.int64)
        cols = np.fromfile(f, it, nz)
        vals = np.fromfile(f, ">f8", nz).astype("<f8")
    rowptr = np.zeros(m + 1, np.int64)
    np.cumsum(rowlen, out=rowptr[1:])
    return sp.csr_matrix((vals, cols.astype(np.int64), rowptr), shape=(m, n))


def read_petsc_vec(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        hdr = np.fromfile(f, ">i4", 2)
        if hdr[0] != PETSC_VEC_ID:
            raise ValueError(f"{path}: not a PETSc Vec binary")
        n = int(hdr[1])
        if path.stat().st_size != 8 + 8 * n:  # 64-bit int header
            f.seek(0)
            n = int(np.fromfile(f, ">i8", 2)[1])
        return np.fromfile(f, ">f8", n).astype("<f8")


def peek_nrows(path: Path) -> int:
    """Number of rows from the header only (MatrixMarket or PETSc binary, 32/64-bit header)."""
    if _detect(path) == "mtx":
        with open(path) as f:
            for line in f:
                if not line.startswith("%"):
                    return int(line.split()[0])
    with open(path, "rb") as f:
        head = f.read(32)
    if int.from_bytes(head[:4], "big") == PETSC_MAT_ID:
        return int.from_bytes(head[4:8], "big")
    return int.from_bytes(head[8:16], "big")


def _detect(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(16)
    if head.startswith(b"%%MatrixMarket"):
        return "mtx"
    if head.startswith(b"SMACSR01") or head.startswith(b"SMAVEC01"):
        return "sma"
    cid = int.from_bytes(head[:4], "big")
    if cid in (PETSC_MAT_ID, PETSC_VEC_ID) or int.from_bytes(head[:8], "big") in (PETSC_MAT_ID, PETSC_VEC_ID):
        return "petsc"
    raise ValueError(f"{path}: unknown format")


def load_matrix(path: Path) -> sp.csr_matrix:
    fmt = _detect(path)
    if fmt == "mtx":
        A = scipy.io.mmread(path)  # multithreaded fast_matrix_market parser (scipy >= 1.12)
        return sp.csr_matrix(A)
    if fmt == "petsc":
        return read_petsc_mat(path)
    return read_csr(path, mmap=False)


def load_rhs(spec: str, n: int) -> np.ndarray:
    if spec == "ones":
        return np.ones(n)
    if spec == "zeros":
        return np.zeros(n)
    path = Path(spec)
    fmt = _detect(path)
    if fmt == "mtx":
        b = scipy.io.mmread(path)
        b = b.toarray() if sp.issparse(b) else np.asarray(b)
        return b.ravel().astype(np.float64)
    if fmt == "petsc":
        return read_petsc_vec(path)
    return read_vec(path)


def _source_ids(matrix: str, rhs: str) -> dict:
    ids = {"source_matrix": str(Path(matrix).resolve()), "source_matrix_sha256": sha256(Path(matrix))}
    if rhs in ("ones", "zeros"):
        ids.update(source_rhs=rhs, source_rhs_sha256=None)
    else:
        ids.update(source_rhs=str(Path(rhs).resolve()), source_rhs_sha256=sha256(Path(rhs)))
    return ids


def up_to_date(out_dir: Path, ids: dict) -> bool:
    """True if out_dir already holds this conversion: same sources (by sha256), same rhs, and
    the binary files still match the checksums recorded at conversion."""
    try:
        meta = json.loads((out_dir / "system.json").read_text())
    except (OSError, ValueError):
        return False
    if any(meta.get(k) != v for k, v in ids.items() if k.endswith("sha256") or k == "source_rhs"):
        return False
    return all((out_dir / f).exists() and sha256(out_dir / f) == h for f, h in meta.get("sha256", {}).items())


def convert(matrix: str, rhs: str, out: str, force: bool = False) -> dict:
    out_dir = Path(out)
    ids = _source_ids(matrix, rhs)
    if not force and up_to_date(out_dir, ids):
        meta = json.loads((out_dir / "system.json").read_text())
        meta["skipped"] = "already converted from the same sources (use --force to redo)"
        return meta
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    A = load_matrix(Path(matrix))
    t_read_src = time.perf_counter() - t0
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix is not square: {A.shape}")
    b = load_rhs(rhs, A.shape[0])
    if b.size != A.shape[0]:
        raise ValueError(f"rhs size {b.size} != nrows {A.shape[0]}")
    t0 = time.perf_counter()
    write_csr(out_dir / "A.csr", A)
    write_vec(out_dir / "b.vec", b)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    A2 = read_csr(out_dir / "A.csr", mmap=False)
    t_read_bin = time.perf_counter() - t0
    meta = {
        **ids,
        "nrows": int(A2.shape[0]),
        "nnz": int(A2.nnz),
        "sha256": {"A.csr": sha256(out_dir / "A.csr"), "b.vec": sha256(out_dir / "b.vec")},
        "timings_s": {"read_source": t_read_src, "write_binary": t_write, "read_binary": t_read_bin},
    }
    (out_dir / "system.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
