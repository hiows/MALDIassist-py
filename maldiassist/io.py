"""Port of R/auxiliary_io.R and R/load_maldi_spectra.R.

Loads Bruker MALDI-TOF ``fid``/``acqu`` files and converts time-of-flight
values to m/z using the Bruker calibration constants (ML1, ML2, ML3).
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from .rcompat import mixedsort


def _search_fid_dir(root: str):
    """Recursively find directories named ``1SLin`` beneath ``root``.

    Mirrors ``list.files(dir, "1SLin", ...)`` which returns full paths whose
    basename matches; the returned order is directory-walk order (sorted per
    directory), matching R's ordering closely enough that name/spectrum pairing
    is preserved because both derive from the same walk.
    """
    hits = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames.sort()
        for d in sorted(dirnames):
            if d == "1SLin":
                hits.append(os.path.join(dirpath, d))
    return hits


def _read_acqu_params(acqu_path: str) -> dict:
    with open(acqu_path, "r", errors="replace") as fh:
        lines = fh.readlines()

    def _grep_value(pattern: str):
        rgx = re.compile(pattern)
        for ln in lines:
            if rgx.search(ln):
                # strip leading up to '= <' and trailing '> '
                val = re.sub(r"(^.*= *<?)|(>? *$)", "", ln.strip("\n"))
                val = val.replace(",", ".")
                return float(val)
        return None

    td = _grep_value(r"##\$TD=")
    delay = _grep_value(r"##\$DELAY=")
    dw = _grep_value(r"##\$DW=")
    ml1 = _grep_value(r"##\$ML1=")
    ml2 = _grep_value(r"##\$ML2=")
    ml3 = _grep_value(r"##\$ML3=")

    number = td
    tof = delay + np.arange(int(number)) * dw

    A = ml3
    B = np.sqrt(1e12 / ml1)
    C = ml2 - tof
    if A == 0:
        mass = (C * C) / (B * B)
    else:
        mass = ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    return {
        "mass": mass,
        "number": int(number),
        "TimeDelay": delay,
        "TimeDelta": dw,
        "tof": tof,
        "c1": ml1,
        "c2": ml2,
        "c3": ml3,
    }


def _load_spectrum(fid_dir: str) -> pd.DataFrame:
    acqu_path = os.path.join(fid_dir, "acqu")
    fid_path = os.path.join(fid_dir, "fid")
    params = _read_acqu_params(acqu_path)
    n = params["number"]
    intensity = np.fromfile(fid_path, dtype="<i4", count=n).astype(float)
    intensity[intensity < 0] = 0.0
    return pd.DataFrame({"mz": params["mass"], "intensity": intensity})


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/\\"))


def _dirname_n(path: str, n: int) -> str:
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def load_maldi_spectra(spectra_dir: str, return_dir: str | None = None) -> "OrderedDict[str, pd.DataFrame]":
    """Load Bruker MALDI-TOF spectra from a directory tree.

    Returns an ordered mapping ``{sample_name: DataFrame(mz, intensity)}`` with
    names derived from the Bruker directory hierarchy and ordered by
    ``mixedsort`` (natural order), exactly as the R implementation.
    """
    spectra_dir = os.path.normpath(os.path.abspath(spectra_dir))

    fid_dirs = _search_fid_dir(spectra_dir)
    fid_dirs = [os.path.normpath(os.path.abspath(p)) for p in fid_dirs]

    raw_spectra = [_load_spectrum(d) for d in fid_dirs]

    # dir_1SLin is the same set of 1SLin directories.
    dir_1slin = list(fid_dirs)
    # base name = basename(dirname^3(dir_1SLin))
    list_names = [_basename(_dirname_n(p, 3)) for p in dir_1slin]

    def _table_counts(names):
        counts = {}
        for nm in names:
            counts[nm] = counts.get(nm, 0) + 1
        return counts

    i = 2
    while any(v > 1 for v in _table_counts(list_names).values()) and i > 0:
        tmp = [_basename(_dirname_n(p, i)) for p in dir_1slin]
        counts = _table_counts(list_names)
        replaced = [nm for nm in sorted(counts) if counts[nm] > 1]
        replaced_set = set(replaced)
        n = len(list_names)
        # R recycles `replaced` across all positions inside paste().
        recycled = [replaced[k % len(replaced)] for k in range(n)] if replaced else [""] * n
        new_names = []
        for k, nm in enumerate(list_names):
            if nm in replaced_set:
                new_names.append(f"{recycled[k]}/{tmp[k]}")
            else:
                new_names.append(nm)
        list_names = new_names
        i -= 1

    named = OrderedDict()
    for nm, spec in zip(list_names, raw_spectra):
        named[nm] = spec

    ordered_names = mixedsort(list(named.keys()))
    result = OrderedDict((nm, named[nm]) for nm in ordered_names)
    return result
