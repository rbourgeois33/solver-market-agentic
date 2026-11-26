"""Raw binary CSR / vector format shared by the Python tools and the C++ drivers.

Little-endian. 128-byte header, then the arrays at the offsets stored in the header
(each 64-byte aligned) so a reader does one fread (or mmap) per array.

Matrix (.csr) header:
  char[8]  magic      "SMACSR01"
  u32      version    1
  u32      idx_bytes  4 or 8   (width of colind entries)
  u64      nrows, ncols, nnz
  u64      off_rowptr, off_colind, off_values
  u32      flags      bit0: rows sorted by column, bit1: duplicates summed
  u32      reserved
  ...      zero padding up to 128 bytes
rowptr is always int64 (nrows+1), values always float64.

Vector (.vec) header:
  char[8]  magic      "SMAVEC01"
  u32      version    1
  u32      reserved
  u64      n
  u64      off_values
  ...      zero padding up to 128 bytes
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HEADER = 128
ALIGN = 64
CSR_MAGIC = b"SMACSR01"
VEC_MAGIC = b"SMAVEC01"
_CSR_FMT = "<8sII3Q3QII"
_VEC_FMT = "<8sIIQQ"
FLAG_SORTED = 1
FLAG_SUMMED = 2


def _align(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


@dataclass
class CsrHeader:
    nrows: int
    ncols: int
    nnz: int
    idx_bytes: int
    off_rowptr: int
    off_colind: int
    off_values: int
    flags: int


def write_csr(path: str | Path, A: sp.csr_matrix) -> None:
    A = sp.csr_matrix(A)
    A.sum_duplicates()  # also sorts indices
    nrows, ncols = A.shape
    nnz = A.nnz
    idx_bytes = 4 if ncols < 2**31 else 8
    idx_dtype = "<i4" if idx_bytes == 4 else "<i8"
    off_rowptr = HEADER
    off_colind = _align(off_rowptr + 8 * (nrows + 1))
    off_values = _align(off_colind + idx_bytes * nnz)
    flags = FLAG_SORTED | FLAG_SUMMED
    hdr = struct.pack(_CSR_FMT, CSR_MAGIC, 1, idx_bytes, nrows, ncols, nnz,
                      off_rowptr, off_colind, off_values, flags, 0)
    with open(path, "wb") as f:
        f.write(hdr.ljust(HEADER, b"\0"))
        A.indptr.astype("<i8", copy=False).tofile(f)
        f.write(b"\0" * (off_colind - f.tell()))
        A.indices.astype(idx_dtype, copy=False).tofile(f)
        f.write(b"\0" * (off_values - f.tell()))
        A.data.astype("<f8", copy=False).tofile(f)


def read_csr_header(path: str | Path) -> CsrHeader:
    with open(path, "rb") as f:
        raw = f.read(HEADER)
    magic, version, idx_bytes, nrows, ncols, nnz, o_r, o_c, o_v, flags, _ = struct.unpack_from(_CSR_FMT, raw)
    if magic != CSR_MAGIC or version != 1:
        raise ValueError(f"{path}: not a SMACSR01 v1 file")
    return CsrHeader(nrows, ncols, nnz, idx_bytes, o_r, o_c, o_v, flags)


def read_csr(path: str | Path, mmap: bool = True) -> sp.csr_matrix:
    h = read_csr_header(path)
    mode = "r" if mmap else None
    idx = "<i4" if h.idx_bytes == 4 else "<i8"
    if mmap:
        rowptr = np.memmap(path, "<i8", mode, h.off_rowptr, (h.nrows + 1,))
        colind = np.memmap(path, idx, mode, h.off_colind, (h.nnz,))
        values = np.memmap(path, "<f8", mode, h.off_values, (h.nnz,))
    else:
        with open(path, "rb") as f:
            f.seek(h.off_rowptr); rowptr = np.fromfile(f, "<i8", h.nrows + 1)
            f.seek(h.off_colind); colind = np.fromfile(f, idx, h.nnz)
            f.seek(h.off_values); values = np.fromfile(f, "<f8", h.nnz)
    return sp.csr_matrix((values, colind, rowptr), shape=(h.nrows, h.ncols), copy=False)


def write_vec(path: str | Path, v: np.ndarray) -> None:
    v = np.ascontiguousarray(v, dtype="<f8").ravel()
    hdr = struct.pack(_VEC_FMT, VEC_MAGIC, 1, 0, v.size, HEADER)
    with open(path, "wb") as f:
        f.write(hdr.ljust(HEADER, b"\0"))
        v.tofile(f)


def read_vec(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read(HEADER)
        magic, version, _, n, off = struct.unpack_from(_VEC_FMT, raw)
        if magic != VEC_MAGIC or version != 1:
            raise ValueError(f"{path}: not a SMAVEC01 v1 file")
        f.seek(off)
        return np.fromfile(f, "<f8", n)
