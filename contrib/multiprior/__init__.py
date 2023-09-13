from copy import deepcopy
import functools as ft
from itertools import product
import time
from pathlib import Path

import einops
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

import src.data
import src.models
import src.utils

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
    inp_path, gt_path, obs_var='five_nadirs', train_domain=None,
):
    """
    Load state-multiprior data.
    """
    inp = xr.open_dataset(inp_path)[obs_var]
    gt = (
        xr.open_dataset(gt_path)
        .ssh.isel(time=slice(0, -1))
        .interp(lat=inp.lat, lon=inp.lon, method='nearest')
    )

    ds =  xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords)

    if train_domain is not None:
        ds = ds.sel(train_domain)

    return xr.Dataset(
        dict(
            input=ds.input,
            tgt=src.utils.remove_nan(ds.tgt),
        ),
        ds.coords,
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
    def test_step(self, batch, batch_idx, latlon=None):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats

        _input = out
        if latlon is not None:
            _input = (out, latlon)

        _priors, _weights = self.solver.prior_cost.detailed_outputs(_input)

        # Make sure priors' outputs and weights have the same channel
        _repeat = lambda x: x
        if out.shape[1] != _weights[0].shape[1]:
            _repeat = lambda x: einops.repeat(
                x, 'b c h w -> b (repeat c) h w', repeat=_priors[0].shape[1],
            )

        _tensors = []
        _tensors.extend([
            batch.input.cpu() * s + m,
            batch.tgt.cpu() * s + m,
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
        legend = ['inp', 'tgt', 'out']  # Somethings is wrong here
        legend.extend([f'phi{k}_out' for k in range(n_priors)])
        legend.extend([f'phi{k}_weight' for k in range(n_priors)])

        self.test_data = rec_da.assign_coords(
            dict(v0=legend)
        ).to_dataset(dim='v0')

        metric_data = self.test_data[legend[:3]].pipe(self.pre_metric_fn)
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
    pass


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

    def forward_ae(self, state):
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, coords = state
        else:  # State
            x, coords = state, state

        phi_outs = torch.stack([phi.forward_ae(x) for phi in self.prior_costs], dim=0)
        phi_weis = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_out = (phi_outs * phi_weis).sum(0)
        return phi_out

    @torch.no_grad()
    def detailed_outputs(self, state):
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, coords = state
        else:  # State
            x, coords = state, state

        phi_outs = [phi.forward_ae(x) for phi in self.prior_costs]
        _weights = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_weis = [_weights[i] for i in range(_weights.shape[0])]

        return phi_outs, phi_weis

    def forward(self, state):
        x = state
        if isinstance(state, (list, tuple)):  # Latitude, longitude
            x, _ = state

        return nn.functional.mse_loss(x, self.forward_ae(state))


class MultiPriorGradSolver(src.models.GradSolver):
    pass


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
