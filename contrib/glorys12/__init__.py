"""
Learning GLORYS12 data
"""
import time
import functools as ft

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from src.data import BaseDataModule, TrainingItem
from src.models import Lit4dVarNet


# Exceptions
# ----------

class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


# Data
# ----

class DistinctNormDataModule(BaseDataModule):
    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]
        normalize = lambda item: (item - m) / s
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
        ])

    def setup(self, stage='test'):
        self.train_ds = LazyXrDataset(
            self.input_da.sel(self.domains['train']),
            **self.xrds_kw, postpro_fn=self.post_fn('train'),
        )
        self.val_ds = LazyXrDataset(
            self.input_da.sel(self.domains['val']),
            **self.xrds_kw, postpro_fn=self.post_fn('val'),
        )


class LazyXrDataset(torch.utils.data.Dataset):
    def __init__(
        self, ds, patch_dims, domain_limits=None, strides=None, postpro_fn=None,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        ds_dims = dict(zip(ds.dims, ds.input.shape))
        self.ds_size = {
            dim: max((ds_dims[dim] - patch_dims[dim]) // 1 + 1, 0)
            for dim in patch_dims
        }

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(),
            np.unravel_index(
                item, tuple(self.ds_size.values())
            )
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        item =  (
            self.ds
            .isel(**sl)
            .to_array()
            .sortby('variable')
        )

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item


# Model
# -----

class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def step(self, batch, phase=""):
        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.rec_weight,
        )

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log(
            f'{phase}_gloss', grad_loss, prog_bar=True, on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase=''):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight, [out, batch.tgt])

        with torch.no_grad():
            self.log(
                f'{phase}_mse', 10000 * loss * self.norm_stats[phase][1]**2,
                prog_bar=True, on_step=False, on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f'{phase}_loss', loss, prog_bar=True, on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return loss, out


# Utils
# -----

def load_glorys12_data(tgt_path, inp_path, tgt_var='zos', inp_var='input'):
    # _isel = dict(time=slice(None, 130))
    _isel = None

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .drop_vars('depth')
        .drop_sel(time=('2012-02-29', '2016-02-29'))
        .isel(_isel)
    )
    inp = xr.open_dataset(inp_path)[inp_var].isel(_isel)

    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)), inp.coords,
        )
        # .to_array()
        # .sortby('variable')
    )


def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    print(f'Durée d\'apprentissage : {time.time() - start:.3} s')
