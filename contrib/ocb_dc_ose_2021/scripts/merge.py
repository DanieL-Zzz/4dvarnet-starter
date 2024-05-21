import hydra
import numpy as np
import pandas as pd
import qf_merge_patches
import xarray as xr


@hydra.main(version_base=None)
def _run(cfg):
    _cfg = dict(cfg)
    resolution = _cfg.pop('resolution')

    # Retrieve patch dimensions
    p = xr.open_dataset(f'{_cfg["input_directory"]}/0.nc')
    time, lat, lon = p.time.shape[0], p.lat.shape[0], p.lon.shape[0]
    def _crop(x):
        return qf_merge_patches.crop(x, crop=int(lat / 12))

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
            time=pd.date_range('2016-12-01', '2018-02-01', freq='1D'),
            lat=np.arange(32., 44., resolution),
            lon=np.arange(-66., -54., resolution),
        ),
    )
    qf_merge_patches.run(**kwargs)


if __name__ == '__main__':
    _run()
