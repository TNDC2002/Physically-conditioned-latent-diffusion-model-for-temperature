"""LMM result keys for multi-checkpoint inference (see ``notebooks/models_inference.ipynb``)."""

from __future__ import annotations

import os
import re


def lmm_checkpoint_tag(ckpt_path: str) -> str:
    """
    Middle segment for ``LMM_PDE_res_<segment>_<n_steps>``:

    - If the filename contains an epoch (``epoch_123``, ``epoch=123``, …): that epoch string
      (digits as in the file, e.g. ``055``).
    - Otherwise: ``last`` (e.g. ``last.ckpt`` or any checkpoint without epoch in the name).
    """
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    m = re.search(r"epoch_(\d+)", stem, re.I) or re.search(r"epoch[=:](\d+)", stem, re.I)
    if m:
        return m.group(1)
    return "last"


def lmm_multickpt_model_key(ckpt_path: str, n_steps: int) -> str:
    """One stored model / timing key: ``LMM_PDE_res_<epoch|last>_<n_steps>``."""
    tag = lmm_checkpoint_tag(ckpt_path)
    return f"LMM_PDE_res_{tag}_{int(n_steps)}"
