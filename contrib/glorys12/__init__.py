"""
Learning GLORYS12 data
"""
import functools as ft
import time
from collections import namedtuple

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr

from src.data import BaseDataModule
from src.models import ConvLstmGradModel, Lit4dVarNet

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt', 'std_'])


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
        return None

    def setup(self, stage='test'):
        self.train_ds = LazyXrDataset(
            self.input_da.sel(self.domains['train']),
            **self.xrds_kw['train'], postpro_fn=self.post_fn('train'),
        )
        self.val_ds = LazyXrDataset(
            self.input_da.sel(self.domains['val']),
            **self.xrds_kw['val'], postpro_fn=self.post_fn('val'),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, batch_size=1, num_workers=1,
        )


class LazyXrDataset(torch.utils.data.Dataset):
    def __init__(
        self, ds, patch_dims, domain_limits=None, strides=None, postpro_fn=None,
        norm_path=None,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ('variable',) + tuple(k for k in self.ds.dims)
        _shape = (2,) + tuple(self.ds[k].shape[0] for k in self.ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        # ds_dims = dict(zip(self.ds.dims, self.ds.shape))
        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }

        self._norm_dims = None
        self._mean, self._std = 0., 1.
        if norm_path:
            norm = xr.open_dataset(norm_path)
            self._norm_dims = tuple(norm.dims)
            self._mean = norm.mean_
            self._std = norm.std_ + 1e-6

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

        item = (
            self.ds
            .isel(**sl)
            # .to_array()
            # .sortby('variable')
        )

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        else:
            norm_sl = sl
            if self._norm_dims:
                norm_sl = {
                    dim: sl[dim] for dim in sl if dim in self._norm_dims
                }
            mean_ = self._mean.isel(**norm_sl).data.astype(np.float32)[None,]
            std_ = self._std.isel(**norm_sl).data.astype(np.float32)[None,]

            item = TrainingItem(
                input=(item[0] - mean_) / std_,
                tgt=(item[1] - mean_) / std_,
                std_=std_,
            )
        return item


# Model
# -----

class NormConvLstmGradModel(ConvLstmGradModel):
    def __init__(self, *args, **kwargs):
        self._patch_dim = kwargs.pop('patch_dim', 48)

        super().__init__(*args, **kwargs)

        self.avgpool = torch.nn.AvgPool2d(kernel_size=self._patch_dim)
        self._c = 0

    def forward(self, x):
        if self._grad_norm is None:
            _shape = x.shape[-2:]

            mode = 'nearest'
            if _shape[0] > self._patch_dim:
                mode = 'bicubic'

            self._grad_norm = (x**2).mean(dim=1, keepdim=True).sqrt() + 1e-6
            self._grad_norm = self.avgpool(self._grad_norm)
            self._grad_norm = torch.nn.functional.interpolate(
                self._grad_norm, size=_shape, mode=mode,
            )

            self._c += 1

            if x.shape[0] > 1 and self._c % 10000 == 0:
                np.save(f'_LOCAL_/tmp/{self._c}_x.npy', x.detach().cpu().numpy(), allow_pickle=True)
                np.save(f'_LOCAL_/tmp/{self._c}_g.npy', self._grad_norm.detach().cpu().numpy(), allow_pickle=True)

        return super().forward(x)


class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        _val_rec_weight = kwargs.pop(
            'val_rec_weight', kwargs['rec_weight'],
        )
        self.train_weights = (
            kwargs.pop('train_weight', 50),
            kwargs.pop('train_weight_grad', 1000),
            kwargs.pop('train_weight_prior', 1.),
        )
        super().__init__(*args, **kwargs)

        self.register_buffer(
            'val_rec_weight',
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == 'val':
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    def on_train_epoch_end(self):
        self.log(
            'n_rejected_batches', self._n_rejected_batches, on_step=False,
            on_epoch=True,
        )

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log(
            f'{phase}_gloss', grad_loss, prog_bar=False, on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = (
            self.train_weights[0] * loss
            + self.train_weights[1] * grad_loss
            + self.train_weights[2] * prior_cost
        )
        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            denormalised_loss = self.weighted_mse(
                (out - batch.tgt) * batch.std_, self.get_rec_weight(phase),
            )

            self.log(
                f'{phase}_mse', 10000 * denormalised_loss,
                prog_bar=True, on_step=False, on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f'{phase}_loss', loss, prog_bar=False, on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            if phase == 'val':
                # Log the loss in Gulfstream
                loss_gf = self.weighted_mse(
                    out[:, :, 445:485, 420:460].detach().cpu().data
                    - batch.tgt[:, :, 445:485, 420:460].detach().cpu().data,
                    np.ones_like(out[:, :, 445:485, 420:460].detach().cpu().data)
                )
                self.log(
                    f'{phase}_loss_gulfstream', loss_gf, on_step=False,
                    on_epoch=True,
                )

        return loss, out


# Utils
# -----

def load_glorys12_data(tgt_path, inp_path, tgt_var='zos', inp_var='input'):
    isel = None  # dict(time=slice(-465, -265))

    _start = time.time()

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
    )
    inp = xr.open_dataset(inp_path)[inp_var].isel(isel)

    ds = (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)), inp.coords,
        )
        .to_array()
        .sortby('variable')
    )

    print(f'>>> Durée de chargement : {time.time() - _start:.4f} s')
    return ds

def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    print(f'Durée d\'apprentissage : {time.time() - start:.3} s')
