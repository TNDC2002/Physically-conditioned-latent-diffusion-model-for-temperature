import torch as th
import torch.nn as nn

from .unet import UNetModel, linear, timestep_embedding


class MFUNet(UNetModel):
    """UNet backbone with separate embeddings for MeanFlow t and (t-r)."""

    def __init__(self, *args, time_scale: float = 1000.0, separate_r_mlp: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_scale = time_scale
        self.separate_r_mlp = separate_r_mlp
        if separate_r_mlp:
            time_embed_dim = self.model_channels * 4
            self.r_time_embed = nn.Sequential(
                linear(self.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.r_time_embed = self.time_embed

    def forward(self, x_t, t, r, context=None):
        hs = []
        t_emb = timestep_embedding(t * self.time_scale, self.model_channels, repeat_only=False)
        # Match MeanFlow paper/reference: second embedding uses delta time (t-r).
        r_emb = timestep_embedding((t - r) * self.time_scale, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb) + self.r_time_embed(r_emb)

        h = x_t.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x_t.dtype)
        return self.out(h)
