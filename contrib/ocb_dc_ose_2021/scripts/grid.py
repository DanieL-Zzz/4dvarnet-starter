import hydra
import numpy as np
import ocn_tools._src.geoprocessing.gridding as ocngrid
import pandas as pd
import xarray as xr


@hydra.main(version_base=None)
def _run(cfg):
    ocngrid.coord_based_to_grid(
        coord_based_ds=xr.open_dataset(cfg["input_path"]),
        target_grid_ds=xr.Dataset(
            coords=dict(
                time=pd.date_range("2016-12-01", "2018-02-01", freq="1D"),
                lat=np.arange(32.0, 44.0, cfg["resolution"]),
                lon=np.arange(-66.0, -54.0, cfg["resolution"]),
            )
        ),
    ).to_netcdf(cfg["output_path"])


if __name__ == "__main__":
    _run()
