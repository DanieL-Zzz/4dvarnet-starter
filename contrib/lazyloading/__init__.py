"""
This extension provides a better handling of massive dataset.
"""
import gc

import numpy as np
import xarray as xr

import src.data
import src.utils

class LazyDataModule(src.data.BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = kwargs.pop('loader')

        # Force the computation of standardisation parameters here
        # It is normally done when the `self.post_fn` is called but it
        # also requires the `self.input_da` to be stored throughout
        # the whole process, which is not in the current situation
        self.input_da = self.loader(domain=self.domains['train'])
        self.norm_stats()
        del self.input_da
        gc.collect()

    def setup(self, stage=None):
        self.train_ds = LazyXrDataset(
            lambda **kwargs: self.loader(domain=self.domains['train'], **kwargs),
            **self.xrds_kw, postpro_fn=self.post_fn(),
        )
        if self.aug_kw:
            self.train_ds = src.data.AugmentedDataset(
                self.train_ds, **self.aug_kw
            )

        self.val_ds = LazyXrDataset(
            lambda **kwargs: self.loader(domain=self.domains['val'], **kwargs),
            **self.xrds_kw, postpro_fn=self.post_fn(),
        )
        self.test_ds = LazyXrDataset(
            lambda **kwargs: self.loader(domain=self.domains['test'], **kwargs),
            **self.xrds_kw, postpro_fn=self.post_fn(),
        )

    def train_mean_std(self, variable='tgt'):
        train_data = (
            self.input_da
            .sel(self.xrds_kw.get('domain_limits', {}))
            .sel(self.domains['train'])
        )
        return (
            train_data[variable]
            .pipe(
                lambda da: (da.mean().item(), da.std().item())
            )
        )


class LazyXrDataset(src.data.XrDataset):
    def __init__(self, loader, **kwargs):
        """
        The `loader` parameter must be a callable which returns a
        xr.DataArray once called.
        """
        self.loader = loader
        super().__init__(self.loader().tgt, **kwargs)

        # Do not store the data during the whole simulation in order to
        # spare memory (it is the point of this whole module).
        del self.da
        gc.collect()

        self.domain_limits = kwargs.get('domain_limits', {})

    def __getitem__(self, item):
        sl = {}
        for dim, idx in zip(
            self.ds_size.keys(),
            np.unravel_index(item, tuple(self.ds_size.values()))
        ):
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim]
            )

        item = (
            self.loader(sl=sl)
            .sel(self.domain_limits or {})

            .to_array()
            .sortby('variable')
        )

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        gc.collect()

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item


def lazily_load_full_natl(
    inp_path, tgt_path, inp_var='five_nadirs', tgt_var='ssh',
    domain=None, sl=None, to_array=False, engine='netcdf4',
):
    """
    Return a xr.Dataset (xr.DataArray if `to_array` is True) instance.

    Following assumptions are made regarding the provided datasets:
    - inp.time.shape[0] = 364 while tgt.time.shape[0] = 365;
    - inp.lat == tgt.lat and inp.lon == tgt.lon;
    - tgt.time values are in incorrect format.

    PARAMETERS
    ----------
    domain: dict
        Filter used to restrict the temporal/spatial data (.sel)

    sl: dict
        Slices indexes used for generating patchs (.isel)

    to_array: bool
        If True, convert the resulting xr.Dataset into xr.DataArray
    """
    with (
        xr.open_dataset(inp_path, engine=engine) as inp,
        xr.open_dataset(tgt_path, engine=engine) as tgt,
    ):
        inp = inp[inp_var]
        tgt = tgt[tgt_var].isel(time=slice(0, -1))
        tgt['time'] = ('time', inp.time.data)

        _filters = lambda da: da.sel(domain or {}).isel(sl or {})

        inp = _filters(inp)
        tgt = _filters(tgt)

        obj = xr.Dataset(
            dict(input=inp, tgt=tgt),
            inp.coords,
        )

        if to_array:
            obj = (
                obj
                .to_array()
                .sortby('variable')
            )

    gc.collect()
    return obj


def crop_one_degree_border(path, dim):
    axis = xr.open_dataset(path)[dim]
    return slice(axis[0]+1, axis[-1]-1)
