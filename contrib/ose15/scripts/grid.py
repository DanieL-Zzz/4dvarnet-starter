import hydra
import numpy as np
import ocn_tools._src.geoprocessing.gridding as ocngrid
import pandas as pd
import xarray as xr


@hydra.main(version_base=None)
def _run(cfg):
    min_time = cfg.get('min_time', '2016-12-01')
    max_time = cfg.get('max_time', '2018-02-01')
    min_lon = cfg.get('min_lon', -66.0)
    max_lon = cfg.get('max_lon', -54.0)
    min_lat = cfg.get('min_lat', 32.0)
    max_lat = cfg.get('max_lat', 44.0)

    ocngrid.coord_based_to_grid(
        coord_based_ds=xr.open_dataset(cfg["input_path"]),
        target_grid_ds=xr.Dataset(
            coords=dict(
                time=pd.date_range(min_time, max_time, freq="1D"),
                lat=np.arange(min_lat, max_lat, cfg["resolution"]),
                lon=np.arange(min_lon, max_lon, cfg["resolution"]),
            )
        ),
    ).to_netcdf(cfg["output_path"])


if __name__ == "__main__":
    _run()
