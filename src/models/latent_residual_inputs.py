"""Shared latent + context construction for residual LDM / LMM (Stage-2 data path).

Keeping this in one module avoids silent drift between ``LatentDiffusion.shared_step``
and ``LatentMeanFlowLitModule.build_latent_and_context``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor


def build_latent_target_and_context_dict(
    autoencoder,
    context_encoder,
    conditional: bool,
    batch: Tuple[Tensor, ...],
) -> Tuple[Tensor, Dict[str, Any]]:
    """Mirror ``LatentDiffusion.shared_step`` up to (but not including) the denoiser forward.

    :param batch: ``(x, y, z, ts)`` — coarse input, HR target, static, time/meta.
    :returns: ``(latent_target, context_dict)`` with ``T_c`` and optional conditioner merge.
    """
    (x, y, z, _ts) = batch

    assert not torch.any(torch.isnan(x)).item(), "coarse input has NaNs"
    assert not torch.any(torch.isnan(y)).item(), "high-res target has NaNs"
    assert not torch.any(torch.isnan(z)).item(), "static has NaNs"

    if autoencoder.ae_flag == "residual":
        residual, _ = autoencoder.preprocess_batch([x, y, z])
        latent_target = autoencoder.encode(residual)[0]
    else:
        latent_target = autoencoder.encode(y)[0]

    context_dict: Dict[str, Any] = {"T_c": x}
    if conditional:
        if context_encoder is None:
            raise ValueError("conditional=True but context_encoder is None")
        encoder_context = context_encoder([(z, [0]), (x, [0])])
        if isinstance(encoder_context, dict):
            context_dict.update(encoder_context)
        else:
            context_dict["encoder_context"] = encoder_context

    return latent_target, context_dict


def context_dict_structure_equal(
    a: Dict[str, Any], b: Dict[str, Any], rtol: float = 0.0, atol: float = 0.0
) -> Tuple[bool, str]:
    """Compare keys and tensor values (``T_c``, cascade maps with tuple keys, etc.)."""
    if set(a.keys()) != set(b.keys()):
        return False, f"key mismatch: {set(a.keys())} vs {set(b.keys())}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                return False, f"shape {k!r}: {tuple(va.shape)} vs {tuple(vb.shape)}"
            if not torch.allclose(va, vb, rtol=rtol, atol=atol):
                d = (va - vb).abs().max().item()
                return False, f"values differ for key {k!r} (max abs {d})"
        elif isinstance(va, dict) and isinstance(vb, dict):
            ok, msg = context_dict_structure_equal(va, vb, rtol=rtol, atol=atol)
            if not ok:
                return False, f"{k!r}.{msg}"
        else:
            return False, f"unsupported or mismatched types for key {k!r}: {type(va)} vs {type(vb)}"
    return True, ""


def context_dict_shapes_match(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """Same keys and tensor shapes (values may differ if encoders differ in init)."""
    if set(a.keys()) != set(b.keys()):
        return False, f"key mismatch: {set(a.keys())} vs {set(b.keys())}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                return False, f"shape {k!r}: {tuple(va.shape)} vs {tuple(vb.shape)}"
        elif isinstance(va, dict) and isinstance(vb, dict):
            ok, msg = context_dict_shapes_match(va, vb)
            if not ok:
                return False, f"{k!r}.{msg}"
        else:
            return False, f"unsupported types for key {k!r}: {type(va)} vs {type(vb)}"
    return True, ""
