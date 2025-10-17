from decimal import Decimal
import argparse
import os
import rasterio
from rasterio.enums import ColorInterp
from django.http import HttpResponse

from rio_tiler.errors import TileOutsideBounds
from rio_tiler.mercator import get_zooms
from rio_tiler import main
from rio_tiler.utils import array_to_image, \
                            get_colormap, \
                            expression, \
                            linear_rescale, \
                            _chunks, \
                            _apply_discrete_colormap, \
                            has_alpha_band, \
                            non_alpha_indexes
from rio_tiler.profiles import img_profiles




from hsvblend import hsv_blend
from formulas import lookup_formula, get_algorithm_list

from rest_framework import exceptions
from django.conf import settings

import numpy as np


ZOOM_EXTRA_LEVELS = 2

def mainTile():
    settings.configure(USE_I18N=False)

    # Argument parser
    parser = argparse.ArgumentParser(description='Return ortophoto tile from image')
    parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
    parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
    parser.add_argument("-d", "--dir", type=str, default='maps', help='File directory')
    parser.add_argument("-f", "--file",    type=str,   default='odm_orthophoto.tif', help='Input file.')
    parser.add_argument("-x", type=float, default='0', help='Image\'s X position.')
    parser.add_argument("-y", type=float, default='0', help='Image\'s Y position.')
    parser.add_argument("-z", type=float, default='0', help='Image\'s Z position.')
    parser.add_argument("-s", "--scale", type=float, default='1', help='Map\'s scale.')
    parser.add_argument("-rs", "--rescale", type=str, default='', help='Map\'s rescale.')
    parser.add_argument("-i", "--index", type=str, default='RGB', help='Map\'s Multispectral Index.')
    parser.add_argument("-for", "--formula", type=str, default='', help='Formula')
    parser.add_argument("-cm", "--color_map", type=str, default='', help='Map Color')
    parser.add_argument("-hs", "--hillshade", type=float, default='0', help='Map Color')
    parser.add_argument("-bnd", "--bands", type=str, default='', help='Map Bands')


    parser.add_argument("-t", "--tile_type", type=str, default='orthophoto', help='User ID')
    args = parser.parse_args()

    # Generate input file
    file = os.path.join(args.base, args.uid, args.dir, "map" + args.index + ".tif")

    print(file)

    """
    Get a tile image
    """
    z = float(args.z)
    x = float(args.x)
    y = float(args.y)
    index = str(args.index)
    tile_type = str(args.tile_type)

    scale = int(args.scale)
    uid = str(args.uid)


    ext = "png"

    indexes = None
    nodata = None
    color_map = None



    formula = str(args.formula)
    bands = str(args.bands)
    rescale = str(args.rescale)
    color_map = str(args.color_map)
    hillshade = float(args.hillshade)

    if formula == '': formula = None
    if bands == '': bands = None
    if rescale == '': rescale = None
    if color_map == '': color_map = None
    if hillshade == '' or hillshade == 0: hillshade = None

    expr = None

    try:
        expr, _ = lookup_formula(formula, bands)
    except ValueError as e:
        raise exceptions.ValidationError(str(e))

    if tile_type in ['dsm', 'dtm'] and rescale is None:
        rescale = "0,1000"

    if tile_type in ['dsm', 'dtm'] and color_map is None:
        color_map = "gray"

    if tile_type == 'orthophoto' and formula is not None:
        if color_map is None:
            color_map = "gray"
        if rescale is None:
            rescale = "-1,1"

    if nodata is not None:
        nodata = np.nan if nodata == "nan" else float(nodata)
    tilesize = scale * 256

    if not os.path.isfile(file):
        print(file)
        print(os.path.isfile(file))
        print(not os.path.isfile(file))
        raise exceptions.NotFound()

    with rasterio.open(file) as src:
        minzoom, maxzoom = get_zoom_safe(src)
        has_alpha = has_alpha_band(src)
        if z < minzoom - ZOOM_EXTRA_LEVELS or z > maxzoom + ZOOM_EXTRA_LEVELS:
            raise exceptions.NotFound()

        # Handle N-bands datasets for orthophotos (not plant health)
        if tile_type == 'orthophoto' and expr is None:
            ci = src.colorinterp

            # More than 4 bands?
            if len(ci) >= 4:

                # Try to find RGBA band order
                if ColorInterp.red in ci and \
                    ColorInterp.green in ci and \
                    ColorInterp.blue in ci:
                    indexes = (ci.index(ColorInterp.red) + 1,
                                ci.index(ColorInterp.green) + 1,
                                ci.index(ColorInterp.blue) + 1,)
                else:
                    # Fallback to first three
                    indexes = (1, 2, 3, )
            elif has_alpha:
                indexes = non_alpha_indexes(src)
        
        # Workaround for https://github.com/OpenDroneMap/WebODM/issues/894
        if nodata is None and tile_type =='orthophoto':
            nodata = 0

    resampling = "nearest"
    padding = 0

    if tile_type in ["dsm", "dtm"]:
        resampling="bilinear"
        padding=16

    try:
        if expr is not None:
            tile, mask = expression(
                file, x, y, z, expr=expr, tilesize=tilesize, nodata=nodata, tile_edge_padding=padding, resampling_method=resampling
            )
        else:
            tile, mask = main.tile(
                file, x, y, z, indexes=indexes, tilesize=tilesize, nodata=nodata, tile_edge_padding=padding, resampling_method=resampling
            )
    except TileOutsideBounds:
        raise exceptions.NotFound("Outside of bounds")

    if color_map:
        try:
            color_map = get_colormap(color_map, format="gdal")
        except FileNotFoundError:
            raise exceptions.ValidationError("Not a valid color_map value")

    intensity = None



    if hillshade is not None:
        try:
            hillshade = float(hillshade)
            if hillshade <= 0:
                hillshade = 1.0
        except ValueError:
            raise exceptions.ValidationError("Invalid hillshade value")

        if tile.shape[0] != 1:
            raise exceptions.ValidationError("Cannot compute hillshade of non-elevation raster (multiple bands found)")

        delta_scale = (maxzoom + ZOOM_EXTRA_LEVELS + 1 - z) * 4
        dx = src.meta["transform"][0] * delta_scale
        dy = -src.meta["transform"][4] * delta_scale

        ls = LightSource(azdeg=315, altdeg=45)

        # Hillshading is not a local tile operation and
        # requires neighbor tiles to be rendered seamlessly
        elevation = get_elevation_tiles(tile[0], file, x, y, z, tilesize, nodata, resampling, padding)
        intensity = ls.hillshade(elevation, dx=dx, dy=dy, vert_exag=hillshade)
        intensity = intensity[tilesize:tilesize*2,tilesize:tilesize*2]


    rgb, rmask = rescale_tile(tile, mask, rescale=rescale)
    rgb = apply_colormap(rgb, color_map)

    if intensity is not None:
        # Quick check
        if rgb.shape[0] != 3:
            raise exceptions.ValidationError("Cannot process tile: intensity image provided, but no RGB data was computed.")

        intensity = intensity * 255.0
        rgb = hsv_blend(rgb, intensity)

    options = img_profiles.get(ext, {})
    buffer = array_to_image(rgb, rmask, img_format=ext, **options)

    sep = "."
    zStr = str(z).split(sep, 1)[0]
    xStr = str(x).split(sep, 1)[0]
    yStr = str(y).split(sep, 1)[0]

    outputFile = os.path.join(args.base, args.uid, args.dir, "output_{0}_{1}_{2}_{3}.{4}".format(zStr, xStr, yStr, index, ext))
    # print(outputFile)

    with open(outputFile, "wb") as f:
        f.write(buffer)
    
    return outputFile



def get_zoom_safe(src_dst):
    minzoom, maxzoom = get_zooms(src_dst)
    if maxzoom < minzoom:
        maxzoom = minzoom

    return minzoom, maxzoom

def rescale_tile(tile, mask, rescale = None):
    if rescale:
        try:
            rescale_arr = list(map(float, rescale.split(",")))
        except ValueError:
            raise exceptions.ValidationError("Invalid rescale value")

        rescale_arr = list(_chunks(rescale_arr, 2))
        if len(rescale_arr) != tile.shape[0]:
            rescale_arr = ((rescale_arr[0]),) * tile.shape[0]

        for bdx in range(tile.shape[0]):
            if mask is not None:
                tile[bdx] = np.where(
                    mask,
                    linear_rescale(
                        tile[bdx], in_range=rescale_arr[bdx], out_range=[0, 255]
                    ),
                    0,
                )
            else:
                tile[bdx] = linear_rescale(
                    tile[bdx], in_range=rescale_arr[bdx], out_range=[0, 255]
                )
        tile = tile.astype(np.uint8)

    return tile, mask

def apply_colormap(tile, color_map = None):
    if color_map is not None and isinstance(color_map, dict):
        tile = _apply_discrete_colormap(tile, color_map)
    elif color_map is not None:
        tile = np.transpose(color_map[tile][0], [2, 0, 1]).astype(np.uint8)

    return tile


if __name__ == "__main__":
    mainTile()