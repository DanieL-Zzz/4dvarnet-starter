import collections
import functools as ft

import einops
import torch
import torch.nn as nn
import xarray as xr

import src.data
import src.models
import src.utils

MultiPriorTrainingItem = collections.namedtuple(
    'MultiPriorTrainingItem', ['input', 'tgt', 'lat', 'lon']
)

def multiprior_entrypoint(trainer, model, dm, test_domain, ckpt=None):
    """
    Specifying the parameter `ckpt` with the path to a checkpoint file
    will trigger the test mode.
    Otherwise, if `ckpt` is not specified, there will be a full training
    and then a testing on the best generated checkpoint.
    """
    print()
    print(trainer.logger.log_dir)
    print()

    if not ckpt:
        trainer.fit(model, dm)
        ckpt = trainer.checkpoint_callback.best_model_path
    src.utils.test_osse(trainer, model, dm, test_domain, ckpt)

def load_data_with_lat_lon(
    inp_path, gt_path, obs_var='five_nadirs', train_domain=None,
):
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


class MultiPriorDataModule(src.data.BaseDataModule):
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


class MultiPriorCost(nn.Module):
    def __init__(self, prior_costs, weight_mod_factory):
        super().__init__()
        self.prior_costs = torch.nn.ModuleList(prior_costs)
        self.weight_mods = torch.nn.ModuleList(
            [weight_mod_factory() for _ in prior_costs]
        )

    def forward_ae(self, state):
        x, coords = state
        phi_outs = torch.stack([phi.forward_ae(x) for phi in self.prior_costs], dim=0)
        phi_weis = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )
        phi_out = ( phi_outs * phi_weis ).sum(0)
        return phi_out

    def forward(self, state):
        return nn.functional.mse_loss(state[0], self.forward_ae(state))


class MultiPriorGradSolver(src.models.GradSolver):
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


class BinWeightMod(nn.Module):
    def forward(self, x, n_prior):
        if n_prior == 0:
            return torch.ones(x.shape[0], device=x.device)[..., None, None, None]

        else:
            return torch.ones(x.shape[0], device=x.device)[..., None, None, None]*float('-inf')


class WeightMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsamp = 5
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )


    def forward(self, x, *args, **kwargs):
        x = einops.reduce(
            x,
            'b c (rlat lat) ( rlon lon) -> b c lat lon',
            'mean',
            rlat=self.downsamp,
            rlon=self.downsamp,
        )
        x = self.net(x)
        x = nn.functional.interpolate(x, scale_factor=self.downsamp, mode='bilinear')
        return x
