"""
Extend the domain along a given dimension in order to make it compatible
with the patch dimensions and the strides of the following condition is
not satisfied:

(total - patch) % stride == 0

Example:

```python
_ds = extend_dim(
    ds, 'latitude', 'constant', patch, stride, constant_values=np.nan,
)
_ds = extend_dim(_ds, 'longitude', 'wrap', patch, stride)
_ds
```
"""

import numpy as np


def extend_dim(ds, dim, mode, patch_size, stride, **kwargs):
    """
    Extend the specified dimension if the provided patch dimensions and
    the stride do not verify the patching condition:

    (total - patch) % stride == 0

    PARAMETERS
    ----------
    ds: xr.Dataset
        Dataset to be modified

    dim: str
        Name of the dimension to extend

    mode: str
        Mode used in the `xr.Dataset.pad` method. For example, it could
        be "constant" or "wrap"

    patch_size: int
        Size of the patch in pixels

    stride: int
        Size of the stride in pixels

    kwargs: dict
        Additional arguments for `xr.Dataset.pad` method

    RETURNS
    -------
    The modified dataset.
    """
    total_length = ds[dim].size

    if (total_length - patch_size) % stride == 0:
        return ds

    n_overlap = (total_length - patch_size) // stride
    total_extension = (n_overlap + 1) * stride + patch_size - total_length
    side_extension = total_extension // 2
    resolution = (ds[dim][1] - ds[dim][0]).item()
    _start, _end = ds[dim][0].item(), ds[dim][-1].item()

    new_coords = np.concat([
        np.arange(
            _start - side_extension * resolution,
            _start,
            resolution,
            dtype=ds[dim].dtype,
        ),
        ds[dim].values,
        np.arange(
            _end + resolution,
            _end + resolution + side_extension * resolution,
            resolution,
            dtype=ds[dim].dtype,
        ),
    ])

    return (
        ds
        .pad(pad_width={dim: int(side_extension)}, mode=mode, **kwargs)
        .assign_coords({dim: new_coords})
    )
