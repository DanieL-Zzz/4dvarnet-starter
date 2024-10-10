import hydra
import numpy as np
import pandas as pd
import qf_merge_patches
import xarray as xr


@hydra.main(version_base=None)
def _run(cfg):
    _cfg = dict(cfg)
    resolution = _cfg.pop('resolution')
    date_min = _cfg.pop('min_time', '2016-12-01')
    date_max = _cfg.pop('max_time', '2018-02-01')
    min_lon = _cfg.pop('min_lon', -66.0)
    max_lon = _cfg.pop('max_lon', -54.0)
    min_lat = _cfg.pop('min_lat', 32.0)
    max_lat = _cfg.pop('max_lat', 44.0)

    # Retrieve patch dimensions
    p = xr.open_dataset(f'{_cfg["input_directory"]}/0.nc')
    time, lat, lon = p.time.shape[0], p.lat.shape[0], p.lon.shape[0]
    def _crop(x):
        return qf_merge_patches.crop(x, crop=int(lat / 17))

    # Merge reconstruction patches
    kwargs = _cfg | dict(
        weight=qf_merge_patches.build_weight(
            patch_dims=dict(time=time, lat=lat, lon=lon),
            dim_weights=dict(
                time=qf_merge_patches.triang,
                lat=_crop,
                lon=_crop,
            )
        ),
        out_coords=dict(
            time=pd.date_range(date_min, date_max, freq='1D'),
            lat=np.arange(min_lat, max_lat, resolution),
            lon=np.arange(min_lon, max_lon, resolution),
        ),
    )
    qf_merge_patches.run(**kwargs)


if __name__ == '__main__':
    _run()
