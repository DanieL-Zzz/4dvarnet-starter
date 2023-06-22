import collections
import functools as ft

import torch
import xarray as xr

import contrib.multiprior
import src.data
import src.models
import src.utils

MultiPriorTrainingItem = collections.namedtuple(
    'MultiPriorTrainingItem', ['input', 'tgt', 'lat', 'lon']
)

def load_data(
    inp_path, gt_path, obs_var='five_nadirs', train_domain=None,
):
    """
    Load (lat, lon)-multiprior data.
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
            latv=ds.lat.broadcast_like(ds.tgt),
            lonv=ds.lon.broadcast_like(ds.tgt),
        ),
        ds.coords,
    ).transpose('time', 'lat', 'lon').to_array()


class MultiPriorLitModel(contrib.multiprior.MultiPriorLitModel):
    def test_step(self, batch, batch_idx):
        super().test_step(
            batch,
            batch_idx,
            latlon=torch.stack((batch.lat[:,0], batch.lon[:,0]), dim=1),
        )


class MultiPriorDataModule(contrib.multiprior.MultiPriorDataModule):
    def get_train_range(self, v):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(
            self.domains['train']
        )
        return train_data[v].min().values.item(), train_data[v].max().values.item()

    def post_fn(self):
        normalize = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        lat_r = self.get_train_range('lat')
        lon_r = self.get_train_range('lon')
        minmax_scale = lambda l, r: 2 * (l - r[0]) / (r[1] - r[0]) - 1.
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiPriorTrainingItem._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
                lambda item: item._replace(lat=minmax_scale(item.lat, lat_r)),
                lambda item: item._replace(lon=minmax_scale(item.lon, lon_r)),
            ],
        )


class MultiPriorGradSolver(contrib.multiprior.MultiPriorGradSolver):
    def init_state(self, batch, x_init=None):
        x_init = super().init_state(batch, x_init)
        coords = torch.stack((batch.lat[:,0], batch.lon[:,0]), dim=1)
        return (x_init, coords)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state[0], batch)
        x, coords = state
        grad = torch.autograd.grad(var_cost, x, create_graph=True)[0]

        x_update = (
            1 / (step + 1) * self.grad_mod(grad)
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        state = (x - x_update, coords)
        return state

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

            if not self.training:
                state = [self.prior_cost.forward_ae(state), state[1]]
        return state[0]
