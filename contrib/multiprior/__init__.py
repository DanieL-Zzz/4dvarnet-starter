from collections import namedtuple
from copy import deepcopy
import functools as ft
from itertools import product
from pathlib import Path
import time

import einops
import kornia.filters as kfilts
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

import src.data
import src.models
import src.utils

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt', 'oi'])


def train(trainer, model, dm, ckpt=None):
    print()
    print('Train and test', trainer.logger.log_dir)
    print()

    _start = time.time()

    trainer.fit(model, dm, ckpt_path=ckpt)
    trainer.test(model, datamodule=dm, ckpt_path='best')

    _duration = time.time() - _start

    print(f'>>> Duration (train+test): {_duration} s')

def test(trainer, model, dm, ckpt):
    print()
    print('Test', trainer.logger.log_dir)
    print()

    trainer.test(model, datamodule=dm, ckpt_path=ckpt)

def load_data(
    inp_path, gt_path, oi_path=None, oi_var='ssh_mod', obs_var='five_nadirs',
    train_domain=None,
):
    """
    Load state-multiprior data.
    """
    train_domain = train_domain or {}

    inp = xr.open_dataset(inp_path)[obs_var].sel(train_domain)
    time_slice = slice(0, -1) if len(inp.time) < 365 else slice(None, None)
    gt = (
        xr.open_dataset(gt_path)
        .ssh
        .sel(train_domain)
        .isel(time=time_slice)
        .interp(lat=inp.lat, lon=inp.lon, method='nearest')
    )
    variables = dict(input=inp, tgt=(gt.dims, gt.values))

    if oi_path:
        oi = (
            xr.open_dataset(oi_path)[oi_var]
            .sel(train_domain)
            .isel(time=time_slice)
            .interp(lat=inp.lat, lon=inp.lon, method='nearest')
        )
        variables['oi'] = (oi.dims, oi.values)

    ds =  xr.Dataset(variables, inp.coords).sel(train_domain)

    ds_variables = dict(input=ds.input, tgt=src.utils.remove_nan(ds.tgt))
    if oi_path:
        ds_variables['oi'] = ds.oi
    return xr.Dataset(
        ds_variables, ds.coords,
    ).transpose('time', 'lat', 'lon').to_array()

def build_domains(domains, train, val, test):
    """
    Add each domain from `domains` to `train`, `val` and `test`. The
    return is a dict of DictConfig.
    """
    result = {
        'train': [],
        'val': [],
        'test': [],
    }
    _items = [('train', train), ('val', val), ('test', test)]

    for (key, oparam), value in product(_items, domains.values()):
        result[key].append(OmegaConf.merge(oparam, value['train']))

    return result

def crop_smallest_containing_domain(subdomains):
    latitude, longitude = [None, None], [None, None]

    for subdomain in subdomains.values():
        start_lat = subdomain['train']['lat'].start
        stop_lat = subdomain['train']['lat'].stop
        start_lon = subdomain['train']['lon'].start
        stop_lon = subdomain['train']['lon'].stop

        if latitude[0] is None or latitude[0] > start_lat:
            latitude[0] = start_lat

        if latitude[1] is None or latitude[1] < stop_lat:
            latitude[1] = stop_lat

        if longitude[0] is None or longitude[0] > start_lon:
            longitude[0] = start_lon

        if longitude[1] is None or longitude[1] < stop_lon:
            longitude[1] = stop_lon

    return {
        'lat': slice(*latitude),
        'lon': slice(*longitude),
    }


class MultiPriorLitModel(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight,
        )
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out), batch)
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def test_step(self, batch, batch_idx, latlon=None):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats
        m_oi, s_oi = self.trainer.datamodule.norm_stats(oi_only=True)

        _input = out
        if latlon is not None:
            _input = (out, latlon)

        _priors, _weights = self.solver.prior_cost.detailed_outputs(_input, batch)

        # Make sure priors' outputs and weights have the same channel
        _repeat = lambda x: x
        if _input.shape[1] != _weights[0].shape[1]:
            _repeat = lambda x: einops.repeat(
                x, 'b c h w -> b (repeat c) h w', repeat=_priors[0].shape[1],
            )

        _tensors = []
        _tensors.extend([
            batch.input.cpu() * s + m,
            batch.tgt.cpu() * s + m,
            batch.oi.cpu() * s_oi + m_oi,
            out.squeeze(dim=-1).detach().cpu() * s + m,
        ])
        _tensors.extend(p.cpu() * s + m for p in _priors)
        _tensors.extend(_repeat(w).cpu() for w in _weights)

        self.test_data.append(torch.stack(_tensors, dim=1))

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )
        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        n_priors = len(self.solver.prior_cost.prior_costs)
        legend = ['inp', 'tgt', 'oi', 'out']  # Somethings is wrong here
        legend.extend([f'phi{k}_out' for k in range(n_priors)])
        legend.extend([f'phi{k}_weight' for k in range(n_priors)])

        self.test_data = rec_da.assign_coords(
            dict(v0=legend)
        ).to_dataset(dim='v0')

        metric_data = self.test_data[legend[:4]].pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data)
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())

        if self.logger:
            (
                self.test_data
                .pipe(self.pre_metric_fn)
                .to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            )
            self.logger.log_metrics(metrics.to_dict())


class MultiPriorDataModule(src.data.BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._norm_stats_oi = None

    def train_mean_std(self, oi_only=False):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        mean_std = lambda var: (
            train_data
            .sel(variable=var)
            .pipe(lambda da: (da.mean().values.item(), da.std().values.item()))
        )

        return mean_std('oi') if oi_only else mean_std('tgt')

    def norm_stats(self, oi_only=False):
        if oi_only:
            if self._norm_stats_oi is None:
                self._norm_stats_oi = self.train_mean_std(oi_only=True)
            return self._norm_stats_oi

        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std(oi_only=oi_only)
        return self._norm_stats

    def post_fn(self):
        m, s = self.norm_stats()
        m_oi, s_oi = self.norm_stats(oi_only=True)

        normalize = lambda item: (item - m) / s
        normalize_oi = lambda item: (item - m_oi) / s_oi

        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
            lambda item: item._replace(oi=normalize_oi(item.oi)),
        ])


class MultiPriorConcatDataModule(src.data.ConcatDataModule):
    def train_mean_std(self):
        sum, count = 0, 0
        for domain in self.domains['train']:
            _sum, _count = (
                self.input_da
                .sel(domain)
                .sel(variable='tgt')
                .pipe(lambda da: (da.sum(), da.pipe(np.isfinite).sum()))
            )
            sum += _sum
            count += _count
        mean = sum / count

        sum = 0
        for domain in self.domains['train']:
            _sum = (
                self.input_da
                .sel(domain)
                .sel(variable='tgt')
                .pipe(lambda da: da - mean)
                .pipe(np.square)
                .sum()
            )
            sum += _sum
        std = (sum / count)**0.5
        return mean.values.item(), std.values.item()

    def setup(self, stage='test'):
        post_fn = self.post_fn()

        _train, _val, _test = [], [], []
        _items = [('train', _train), ('val', _val), ('test', _test)]
        for key, _list in _items:
            _list.extend(
                src.data.XrDataset(
                    self.input_da.sel(domain),
                    **self.xrds_kw,
                    postpro_fn=post_fn,
                )
                for domain in self.domains[key]
            )

        self.train_ds = src.data.XrConcatDataset(_train)
        if self.aug_kw:
            self.train_ds = src.data.AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = src.data.XrConcatDataset(_val)
        self.test_ds = src.data.XrConcatDataset(_test)


class MultiPriorCost(nn.Module):
    def __init__(self, prior_costs, weight_mod_factory):
        super().__init__()
        self.prior_costs = torch.nn.ModuleList(prior_costs)
        self.weight_mods = torch.nn.ModuleList(
            [weight_mod_factory() for _ in prior_costs]
        )

    def forward_ae(self, state, batch):
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, coords = state
        else:  # State
            x, coords = state, batch.oi.nan_to_num()

        phi_outs = torch.stack([phi.forward_ae(x) for phi in self.prior_costs], dim=0)
        phi_weis = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_out = (phi_outs * phi_weis).sum(0)
        return phi_out

    @torch.no_grad()
    def detailed_outputs(self, state, batch):
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, coords = state
        else:  # State
            x, coords = state, batch.oi.nan_to_num()

        phi_outs = [phi.forward_ae(x) for phi in self.prior_costs]
        _weights = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_weis = [_weights[i] for i in range(_weights.shape[0])]

        return phi_outs, phi_weis

    def forward(self, state, batch):
        x = state
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, _ = state

        return nn.functional.mse_loss(x, self.forward_ae(state, batch))


class MultiPriorGradSolver(src.models.GradSolver):
    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state, batch) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state, batch)
        return state


class WeightMod(nn.Module):
    def __init__(self, in_channel, out_channel, upsample_mode, resize_factor=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(resize_factor),
            nn.Conv2d(in_channel, out_channel, 7, padding=3),
            nn.Upsample(scale_factor=resize_factor, mode=upsample_mode),

            nn.Sigmoid(),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)
