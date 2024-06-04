"""
Trying to learn a scale-equivariant model f, i. e. f(ax) = af(x).
"""

import torch

from src.models import Lit4dVarNet


class LitEquivariant4dVarNet(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        self.bounds = kwargs.pop("bounds", [1.5, 0.5])
        super().__init__(*args, **kwargs)

    def base_step(self, batch, phase=""):
        if phase == "train":
            a = (  # random float for each state
                (self.bounds[1] - self.bounds[0])
                * torch.rand(
                    batch.input.shape[0],
                    requires_grad=True,
                    device=batch.input.device,
                )
                + self.bounds[0]
            ).reshape(-1, 1, 1, 1)

            abatch = batch._replace(input=a * batch.input)
            out = self(batch=abatch) / a
        else:
            out = self(batch=batch)

        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(
                f"{phase}_mse", 10000 * loss * self.norm_stats[1] ** 2,
                prog_bar=True, on_step=False, on_epoch=True,
            )
            self.log(
                f"{phase}_loss", loss,
                prog_bar=True, on_step=False, on_epoch=True,
            )

        return loss, out
