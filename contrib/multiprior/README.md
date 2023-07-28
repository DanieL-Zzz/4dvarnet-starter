# Multiprior

## Configurations

Configuration files (yaml) are located in:

```
/config/xp/mp/
```

As 28 july 2023, it contains the following experiment files:

| Filename | Specificity |
| -------- | ----------- |
| starter-0 | Basic version of multiprior (2 priors). The weight architecture is just a AvgPool2d > Conv2d > Upsampling > Sigmoid |
| starter-1 | Same as starter-0 but on two different domains (concatenated) |
| starter-2 | Same as starter-0 but the output of the Weight network has the same number of channel as the input |
| starter-3 | Add a deeper Weight network and the possibility to choose the different weights networks. Plus, we are trying to feed it with the OI or the SST data (OI is not working for now) |

**starter-0** to **starter-2** (included) can be run from the Git branch `multiprior` while **starter-3** can only be run from the Git branch `mp-oi` for the moment.

By default, the Weights networks are fed with the state. If you want to
feed them with the latitude and longitude, replace the key configuration `_scheme`:

```yaml
_scheme: ${_sch.state}  # for state as input
_scheme: ${_sch.latlon}  # for lat and lon as input
```

To modify the parameters of the priors or the Weights networks, see the
key configuration `model.solver.prior_cost` and `_sch.{state|latlon}.weight` if necessary.

The branch `multiprior` does not allow different Weights networks
parameters yet (key config `model.solver.prior_cost.weight_mod_factory` refers directly
to a Weight network class). In the branch `mp-oi` (ongoing), it is possible to
specify a list of different Weight network class.

In **starter-3**, feeding the Weight network with OI or SST is configurable with
the key configuration `datamodule.input_da.oi_path` and `datamodule.input_da.oi_var`
(the name contains "oi" but we can give them SST path and variable, I haven't changed
the name yet).

A file `/config/_LOCAL_imt.yaml` is required and must contain the paths
to the datasets. In IMT servers for example:

```yaml
path:  # IMT server local paths
  input: '/DATASET/NATL/cal_data_new_errs.nc'  # five nadirs
  gt: '/DATASET/NATL/NATL60-CJM165_NATL_ssh_y2013.1y.nc'
  oi: '/DATASET/NATL/ssh_NATL60_swot_4nadir.nc'
  sst: '/DATASET/NATL/NATL60-CJM165_NATL_sst_y2013.1y.nc'
```

## Source code

The codes are available at `/contrib/multiprior/`. The `__init__.py` file contains
the general source code of the multiprior, the specificity of the latitude and
longitude scheme is handled in t he file `latlon.py`.

All classes inherit from the files `/src/models.py` and `/src/data.py`.

In `__init__.py` from branch `multiprior`:

- Functions `train` and `test` are related to the training and testing;
- `load_data` retrieves and crops the desired data;
- `build_domains` and `crop_smallest_containing_domain` are related to the
    multi-domains case;
- In the class `MultiPriorLitModel`, we override the `src.models.Lit4DVarNet`'s
    test step and at the end of the tests, we store the reconstruction and the
    intermediate priors and their respective Weights networks in a netCDF4
    dataset (`test_data.nc`);
- In `MultiPriorConcatDataModule`, we override `src.data.ConcatDataModule` so
    the multi-domains scheme is handled by multiprior;
- In `MultiPriorCost`, it's the multiprior. The method `detailed_outputs` is
    called at the end of tests for generating intermediate priors and Weights
    networks values inside the netCDF4 file;
- The Weight network is defined in the class `WeightMod`. In branch `mp-oi`, there
    is an additional Weight network class called `DeeperWeightMod` which is just
    a little bit deeper version of `WeightMod`, nothing more (for now).


## Run

To start a training:

```
python main.py xp=mp/starter-X
```

where `X` is the multiprior experiment number you want (see in `/config/xp/mp`).

If the training stopped and you want to resume the training from the
last available checkpoint file `Y`:

```
python main.py xp=mp/starter-X _checkpoint=Y
```

To start a test with the `Y` checkpoint:

```
python main.py xp=mp/starter-X _entrypoint=test _checkpoint=Y
```


## Todolist

Ronan asked me to reduce the level of detail in the weight network (this seems
to be doable by increasing the Weight network's `resize_factor` parameter).

Also, I have to feed the Weights networks with OI data but I struggle to load
them properly (almost blank maps...), so I started with SST data.

Finally, Ronan asked me to run the multiprior on multidomains with Lat, Lon. I
haven't done it yet, I suppose some fixes are required in the `latlon.py` as I
haven't touched it for a while.

I'll resume everything on september.