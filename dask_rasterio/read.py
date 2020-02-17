import rasterio
import numpy as np
import dask.array as da
from dask.base import tokenize
from rasterio.windows import Window


def read_raster(path, bands=None, masked=False, block_size=1):
    """
    Read all or some band_ids from raster

    Arguments:
        path {string} -- path to raster file

    Keyword Arguments:
        bands {int, iterable(int)} -- bands number or iterable of band_ids.
            When passing None, it reads all band_ids (default: {None})
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
        # index of min(data_set.block_shapes)
        return [
            (pos, resize_window(win, blk_size))
            for pos, win in data_set.block_windows(bidx=band_id)
        ]

    assert isinstance(path, (str, pathlib.Path))
    if isinstance(path, str):
        path = pathlib.Path(path)
    with rasterio.open(path) as src:
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
        name = 'Raster-{}'.format(tokenize(path.absolute(), bands, chunks))
        if isinstance(bands, (tuple, list)):
            shape = len(bands), *src.shape
            dsk = {
                (name, 0, i, j): (read_window, path, window, bands, masked)
                for (i, j), window in blocks
            }
        else:
            shape = src.shape
            dsk = {
                (name, i, j): (read_window, path, window, bands, masked)
                for (i, j), window in blocks
            }
        # return dsk, name, chunks, dtype, shape
        return da.Array(
            dask=dsk,
            name=name,
            chunks=chunks,
            dtype=dtype,
            shape=shape
        )


def read_raster_band(path, band=1, masked=False, block_size=1):
    """
    Read a raster band_id and return a Dask array

    Arguments:
        path {string} -- path to the raster file

    Keyword Arguments:
        band_id {int} -- number of band_id to read (default: {1})
        blk_size {int} -- block size multiplier (default: {1})

    """

    def read_window(raster_path, window, band_id):
        with rasterio.open(raster_path) as src_path:
            return src_path.read(band_id, window=window, masked=masked)

    def resize_window(window, blk_size):
        return Window(
            col_off=window.col_off * blk_size,
            row_off=window.row_off * blk_size,
            width=window.width * blk_size,
            height=window.height * blk_size)

    def block_windows(data_set, band_id, blk_size):
        return [(pos, resize_window(win, blk_size))
                for pos, win in data_set.block_windows(band_id)]

    with rasterio.open(path) as src:
        h, w = src.block_shapes[band - 1]
        chunks = (h * block_size, w * block_size)
        name = 'raster-{}'.format(tokenize(path, band, chunks))
        dtype = src.dtypes[band - 1]
        shape = src.shape
        blocks = block_windows(src, band, block_size)

    dsk = {(name, i, j): (read_window, path, window, band)
           for (i, j), window in blocks}

    return da.Array(dsk, name, chunks, dtype, shape)


def get_band_count(raster_path):
    """Read raster bands count"""
    with rasterio.open(raster_path) as src:
        return src.count
