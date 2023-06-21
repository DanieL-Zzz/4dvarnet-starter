import collections
from pathlib import Path

import einops
import pandas as pd
import torch
import torch.nn as nn

import src.data
import src.models
import src.utils

def train(trainer, model, dm, ckpt=None):
    print()
    print('Train', trainer.logger.log_dir)
    print()

    trainer.fit(model, dm, ckpt_path=ckpt)
    trainer.test(model, datamodule=dm, ckpt_path='best')

def test(trainer, model, dm, ckpt):
    print()
    print('Train', trainer.logger.log_dir)
    print()

    trainer.test(model, datamodule=dm, ckpt_path=ckpt)


class MultiPriorLitModel(src.models.Lit4dVarNet):
    def test_step(self, batch, batch_idx, latlon=None):
        """
        If `_input` is specified, add its value as input to MultiPrior
        along with the output of the solver.

        Used solely for latitude and longitude.
        """
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats

        _input = out
        if latlon:
            _input = (out, latlon)

        _priors, _weights = self.solver.prior_cost.detailed_outputs(_input)

        # Make sure priors' outputs and weights have the same channel
        _repeat = lambda x: einops.repeat(
            x, 'b c h w -> b (repeat c) h w', repeat=_priors[0].shape[1],
        )

        _tensors = []
        _tensors.extend([
            batch.input.cpu() * s + m,
            batch.tgt.cpu() * s + m,
            out.squeeze(dim=-1).detach().cpu() * s + m,
        ])
        _tensors.extend(p.cpu() for p in _priors)
        _tensors.extend(_repeat(w).cpu() for w in _weights)

        self.test_data.append(torch.stack(_tensors, dim=1))

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )
        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        n_priors = len(self.solver.prior_cost.prior_costs)
        legend = ["obs", "ssh", "out"]
        legend.extend([f'phi{k}_out' for k in range(n_priors)])
        legend.extend([f'phi{k}_weight' for k in range(n_priors)])

        self.test_data = rec_da.assign_coords(
            dict(v0=legend)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data)
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())

        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())


class MultiPriorDataModule(src.data.BaseDataModule):
    pass


class MultiPriorCost(nn.Module):
    def __init__(self, prior_costs, weight_mod_factory):
        super().__init__()
        self.prior_costs = torch.nn.ModuleList(prior_costs)
        self.weight_mods = torch.nn.ModuleList(
            [weight_mod_factory() for _ in prior_costs]
        )

    def forward_ae(self, state):
        if isinstance(state, collections.abc.Iterable):  # Latitude, longitude
            x, coords = state
        else:
            x, coords = state, state

        phi_outs = torch.stack([phi.forward_ae(x) for phi in self.prior_costs], dim=0)
        phi_weis = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_out = (phi_outs * phi_weis).sum(0)
        return phi_out

    @torch.no_grad()
    def detailed_outputs(self, state):
        if isinstance(state, collections.abc.Iterable):  # Latitude, longitude
            x, coords = state
        else:
            x, coords = state, state

        phi_outs = [phi.forward_ae(x) for phi in self.prior_costs]
        _weights = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_weis = [_weights[i] for i in range(_weights.shape[0])]

        return phi_outs, phi_weis

    def forward(self, state):
        return nn.functional.mse_loss(state[0], self.forward_ae(state))


class MultiPriorGradSolver(src.models.GradSolver):
    pass


class WeightMod(nn.Module):
    def __init__(self, resize_factor=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(resize_factor),
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Upsample(scale_factor=resize_factor, mode='bilinear'),

            nn.Sigmoid(),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)
