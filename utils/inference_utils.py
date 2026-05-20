from __future__ import annotations

import torch


def lmm_uniform_tr_schedule(n_steps: int, device: torch.device, dtype: torch.dtype) -> list[tuple[float, float]]:
    """Uniform knots from t=1 to 0: ``linspace(1, 0, n_steps+1)``."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    t_vals = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device, dtype=dtype)
    return [(float(t_vals[i].item()), float(t_vals[i + 1].item())) for i in range(n_steps)]


def lmm_inv_ni_tr_schedule(n_steps: int, device: torch.device, dtype: torch.dtype) -> list[tuple[float, float]]:
    """Reciprocal warp: for 1-indexed step i, r_i = 1/(n_steps * i); t_1=1, t_{i>1}=r_{i-1}; r_n=0.

    Example (n_steps=3): (1, 1/3), (1/3, 1/6), (1/6, 0).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    pairs: list[tuple[float, float]] = []
    for k in range(n_steps):
        t_val = 1.0 if k == 0 else 1.0 / (3 * k)
        r_val = 0.0 if k == n_steps - 1 else 1.0 / (3 * (k + 1))
        pairs.append((t_val, r_val))
    return pairs


@torch.no_grad()
def _lmm_predict_final_n_steps_scheduled(
    model,
    low_res: torch.Tensor,
    static: torch.Tensor,
    y_hr: torch.Tensor,
    tr_pairs: list[tuple[float, float]],
) -> torch.Tensor:
    """Multi-step latent MeanFlow inference with explicit (t_i, r_i) pairs."""
    residual, _ = model.autoencoder.preprocess_batch([low_res, y_hr, static])
    z_enc = model.autoencoder.encode(residual)[0]
    context = model._build_context_dict(low_res, static)

    x_cur = torch.randn_like(z_enc)
    batch_size = x_cur.shape[0]
    step_dtype = low_res.dtype
    dev = low_res.device

    for t_val, r_val in tr_pairs:
        t_i = torch.full((batch_size,), t_val, device=dev, dtype=step_dtype)
        r_i = torch.full((batch_size,), r_val, device=dev, dtype=step_dtype)
        u_theta = model.mf_unet(x_cur, t_i, r_i, context=context)
        x_cur = model.meanflow_core.single_step_generate(x_cur, t_i, r_i, u_theta)

    r_hat = model.autoencoder.decode(x_cur)
    merged = model.autoencoder.nn_lr_and_merge_with_static(low_res, static)
    y_up = model.autoencoder.unet(merged)
    return y_up + r_hat


@torch.no_grad()
def lmm_predict_final_n_steps_uniform(
    model,
    low_res: torch.Tensor,
    static: torch.Tensor,
    y_hr: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Multi-step MeanFlow inference with a uniform time grid (t=1 .. 0)."""
    pairs = lmm_uniform_tr_schedule(n_steps, low_res.device, low_res.dtype)
    return _lmm_predict_final_n_steps_scheduled(model, low_res, static, y_hr, pairs)


@torch.no_grad()
def lmm_predict_final_n_steps_inv_ni(
    model,
    low_res: torch.Tensor,
    static: torch.Tensor,
    y_hr: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Multi-step MeanFlow inference with warp r_i = 1/(n_steps * i) (i 1-indexed); final r=0."""
    pairs = lmm_inv_ni_tr_schedule(n_steps, low_res.device, low_res.dtype)
    return _lmm_predict_final_n_steps_scheduled(model, low_res, static, y_hr, pairs)


def get_model_output(model_type, model, loaded_data, sampler = None, num_diffusion_iters = None):
    if model_type == "unet-like":
        # ``DownscalingDataset``: ``nn_lowres=True`` embeds static into ``low_res`` on the CPU;
        # ``nn_lowres=False`` returns ``(lr, hr, static, time)``. Match training / LMM / LDM by
        # upsampling ``lr`` and concatenating ``static`` on GPU (``UnetLitModule._merge_lr_and_static``).
        with torch.no_grad():
            if len(loaded_data) == 4 and hasattr(model, "_merge_lr_and_static"):
                lr, _hr, static, _ts = loaded_data
                x = model._merge_lr_and_static(
                    lr.to(device="cuda:0"), static.to(device="cuda:0")
                )
                test1 = model(x).cpu()
            else:
                test1 = model(loaded_data[0].to(device="cuda:0")).cpu()
        ts_ns = loaded_data[-1]
        return test1, ts_ns
    elif model_type=='ldm':
        low_res = loaded_data[0]     
        static = loaded_data[2]
        # Generate residual and endode it!
        with torch.no_grad():
            residual, _ = model.autoencoder.preprocess_batch([ld.to(device='cuda:0') for ld in loaded_data[:-1]])
            high_res_encoded = model.autoencoder.encode(residual.to(device='cuda:0'))[0]

        gen_shape = tuple(high_res_encoded.shape[1::])
        # Run ldm model to get estimate of high-res in latent space
        timesteps = torch.arange(0, 1, dtype=static.dtype).unsqueeze(0).expand(static.shape[0],-1)
        with torch.no_grad():
            ext_context = [[static.to(device='cuda:0'),timesteps],
                           [low_res.to(device='cuda:0'),timesteps]]
            test1 = sampler.run_ldm_sampler(ext_context, num_diffusion_iters, 1, gen_shape)
        # Run decoder to get estimate of high-res in pixel space
        with torch.no_grad():
            decoded_data = model.autoencoder.decode(test1).cpu()
        # Get reference timestep
        ts_ns = loaded_data[3]
        if model.autoencoder.ae_flag == 'residual':
            # Add back the unet results to the decoded residual
            low_res_nn_merged_with_satic = model.autoencoder.nn_lr_and_merge_with_static(loaded_data[0],loaded_data[2])
            with torch.no_grad():
                result = decoded_data + model.autoencoder.unet(low_res_nn_merged_with_satic.to(device='cuda:0')).cpu()
            return result, ts_ns
        else:        
            return decoded_data, ts_ns
    elif model_type == 'meanflow-residual':
        low_res = loaded_data[0].to(device='cuda:0')
        static = loaded_data[2].to(device='cuda:0')
        ts_ns = loaded_data[3]
        with torch.no_grad():
            pred_final = model.predict_final(
                low_res=low_res,
                static=static,
                y_shape=(low_res.shape[0], 1, static.shape[-2], static.shape[-1]),
            ).cpu()
        return pred_final, ts_ns
    elif model_type == 'lmm':
        dev = next(model.parameters()).device
        low_res = loaded_data[0].to(dev)
        y_hr = loaded_data[1].to(dev)
        static = loaded_data[2].to(dev)
        ts_ns = loaded_data[3]
        with torch.no_grad():
            pred_final = model.predict_final(low_res=low_res, static=static, y_hr=y_hr).cpu()
        return pred_final, ts_ns