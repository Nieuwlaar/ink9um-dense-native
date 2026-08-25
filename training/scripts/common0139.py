"""Shim so the label-build scripts resolve their one external helper.

`calibrate_theta.py` and `build_dense7_labels.py` import `roc_auc` from a
`common0139` module. It is the same tie-corrected Mann-Whitney ROC-AUC that
ships in this repo's `native-eval/native_bench.py`, so this file just
re-exports it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "native-eval"))

from native_bench import roc_auc  # noqa: E402,F401

__all__ = ["roc_auc"]
