import rasterio
import numpy as np
import dask.array as da
from dask.base import tokenize
from rasterio.windows import Window


def read_raster(image_path, bands=None, masked=False, block_size=1):
    """
    Read all or some band_ids from raster

    Arguments:
        image_path {string} -- image_path to raster file

    Keyword Arguments:
        bands {int, iterable(int)} -- bands number or iterable of band_ids.
            When passing None, it reads all band_ids (default: {None})
        masked {bool} -- If `True`, returns masked array masking `nodata` values
            (default: {False})
        block_size {int} -- block size multiplier (default: {1})

    Returns:
        dask.array.Array -- a Dask array
    """
    import pathlib

    def read_window(raster_path, window, band_ids=None, if_masked=False):
        with rasterio.open(raster_path) as src_path:
            return src_path.read(
                indexes=band_ids, window=window, masked=if_masked
            )

    def resize_window(window, blk_size):
        return Window(
            col_off=window.col_off * blk_size,
            row_off=window.row_off * blk_size,
            width=window.width * blk_size,
            height=window.height * blk_size)

    def block_windows(data_set, blk_size):
        shape_list = data_set.block_shapes
        band_id = shape_list.index(min(shape_list)) + 1
        return [
            (pos, resize_window(win, blk_size))
            for pos, win in data_set.block_windows(bidx=band_id)
        ]

    assert isinstance(image_path, (str, pathlib.Path))
    if isinstance(image_path, str):
        image_path = pathlib.Path(image_path)
    with rasterio.open(image_path) as src:
        if bands is None:
            bands = list(range(1, (src.count + 1)))
        b_shapes = np.array(src.block_shapes)
        h, w = np.min(b_shapes[:, 0]), np.min(b_shapes[:, 1])
        u_dtypes = set(src.dtypes)
    assert isinstance(bands, (int, tuple, list))
    if isinstance(bands, int):
        chunks = (h * block_size, w * block_size)
    else:
        chunks = (len(bands), h * block_size, w * block_size)
    if len(u_dtypes) > 1:
        raise ValueError(
            "Multiple 'dtype' found.\n** Read individual band instead! **"
        )
    else:
        assert len(
            u_dtypes
        ) == 1, "No 'dtype' found!\n** Possibly corrupted File **"
        dtype = u_dtypes.pop()
        blocks = block_windows(src, block_size)
        name = 'Raster-{}'.format(
            tokenize(image_path.absolute(), bands, chunks)
        )
        if isinstance(bands, (tuple, list)):
            shape = len(bands), *src.shape
            dsk = {
                (name, 0, i, j): (
                    read_window, image_path, window, bands, masked
                )
                for (i, j), window in blocks
            }
        else:
            shape = src.shape
            dsk = {
                (name, i, j): (read_window, image_path, window, bands, masked)
                for (i, j), window in blocks
            }
        return da.Array(
            dask=dsk,
            name=name,
            chunks=chunks,
            dtype=dtype,
            shape=shape
        )
