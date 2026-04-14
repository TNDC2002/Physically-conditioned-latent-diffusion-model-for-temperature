import torch


@torch.no_grad()
def generate_residual_one_step(mf_unet, meanflow_core, context, shape, device, dtype=torch.float32):
    batch_size = shape[0]
    x_t = torch.randn(shape, device=device, dtype=dtype)
    t = torch.ones(batch_size, device=device, dtype=dtype)
    r = torch.zeros(batch_size, device=device, dtype=dtype)
    u_theta = mf_unet(x_t, t, r, context=context)
    return meanflow_core.single_step_generate(x_t, t, r, u_theta)
