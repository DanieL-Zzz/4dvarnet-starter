"""
This contribution implements the solver as it was first implemented in
the repository 4dvarnet-core.
"""

import torch

from src.models import GradSolver, Lit4dVarNet


class LitCoreVarNet(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        self.n_repeat = kwargs.pop('n_repeat', 1)
        super().__init__(*args, **kwargs)

    def forward(self, batch, state):
        return self.solver(batch, state)

    def base_step(self, batch, phase=""):
        losses = []
        state = None

        for _ in range(self.n_repeat):
            out = self(batch=batch, state=state)
            losses.append(
                self.weighted_mse(out - batch.tgt, self.rec_weight)
            )
            state = out.detach().requires_grad_(True)

        if self.training:
            loss = torch.stack(losses).sum()
        else:
            loss = losses[-1]

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[1] ** 2,
                prog_bar=True, on_step=False, on_epoch=True,
            )
            self.log(
                f"{phase}_loss", loss,
                prog_bar=True, on_step=False, on_epoch=True,
            )

        return loss, out


class CoreGradSolver(GradSolver):
    @torch.enable_grad()
    def forward(self, batch, state=None):
        if state is None:
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

        for step in range(self.n_step):
            state = self.solver_step(state, batch, step=step)

            if not self.training:
                state = state.detach().requires_grad_(True)

        if not self.training:
            state = self.prior_cost.forward_ae(state)

        return state
