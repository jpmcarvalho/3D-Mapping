#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fábio Azevedo
"""

from osgeo import gdal
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import exiftool

import sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages/')

import cv2
import shutil
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Argument parser
parser = argparse.ArgumentParser(description='Generate NDVI from odm_orthophoto.')
parser.add_argument("-b", "--base",    type=str,   default='/code/data',         help='Base directory')
parser.add_argument("-d", "--dir",     type=str,   default='odm_orthophoto',     help='File directory')
parser.add_argument("-f", "--file",    type=str,   default='odm_orthophoto.tif', help='Input file.')
parser.add_argument("-O", "--outdir",  type=str,   default='maps',               help='Output directory')
parser.add_argument("-o", "--outfile", type=str,   default='map',                help='Output file.')
parser.add_argument("-u", "--uid",     type=str,   default='1',                  help='User ID')
parser.add_argument("-g", "--gamma",   type=float, default='1.2',                help='Gamma Correction factor')
parser.add_argument(      "--ndvi", action="store_true", help='NDVI.')
parser.add_argument(      "--rgb",  action="store_true", help='RGB.')
parser.add_argument(      "--ndre", action="store_true", help='NDRE.')
parser.add_argument(      "--cir",  action="store_true", help='CIR.')
parser.add_argument(      "--ndwi", action="store_true", help='NDWI.')
parser.add_argument(      "--thermal", action="store_true", help='THERMAL.')
parser.add_argument(      "--flir", action="store_true", help='FLIR.')
parser.add_argument(      "--isMultispectral", help="Post-process Multispectral images", action="store_true")
parser.add_argument("-v", "--version",         help="show program version",              action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

# Generate input file and output directory path strings
file   = os.path.join(args.base, args.uid, args.dir, args.file)
outdir = os.path.join(args.base, args.uid, args.outdir)

if not os.path.exists(file):
    print("File " + file + " not found!")
    exit()

os.makedirs(outdir, 0o777, True)

def doGrayMap(name, grayMap, alpha=None):

    if alpha is None:
        alpha = np.zeros(grayMap.shape, dtype=grayMap.dtype)
        alpha[grayMap != 0] = 255

    [cols, rows] = grayMap.shape

    outFileNameGray = os.path.join(outdir, args.outfile + str(name) + "Raw.tif")
    outdata = driver.Create(outFileNameGray, rows, cols, 4, gdal.GDT_Byte, options=options)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(grayMap)
    outdata.GetRasterBand(2).WriteArray(grayMap)
    outdata.GetRasterBand(3).WriteArray(grayMap)
    outdata.GetRasterBand(4).WriteArray(alpha)
    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    gray = cv2.imread(outFileNameGray, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(os.path.join(outdir, args.outfile + str(name) + "Raw.png"), gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    outFileNameGrayColored = os.path.join(outdir, args.outfile + str(name) + ".tif")
    outdata = driver.Create(outFileNameGrayColored, rows, cols, 4, gdal.GDT_Byte, options=options)

    imgResultColor = cv2.applyColorMap(grayMap.astype('uint8'), cv2.COLORMAP_JET)

    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(imgResultColor[:, :, 2])
    outdata.GetRasterBand(2).WriteArray(imgResultColor[:, :, 1])
    outdata.GetRasterBand(3).WriteArray(imgResultColor[:, :, 0])
    outdata.GetRasterBand(4).WriteArray(alpha)
    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!
    outdata = None

    grayColored = cv2.imread(outFileNameGrayColored, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(os.path.join(outdir, args.outfile + str(name) + ".png"), grayColored, [cv2.IMWRITE_PNG_COMPRESSION, 9])

if os.environ.get('exiftoolpath') is not None:
    exiftoolPath = os.path.normpath(os.environ.get('exiftoolpath'))
else:
    exiftoolPath = None
with exiftool.ExifTool(exiftoolPath) as exift:
    exifData = exift.get_metadata(file)

ds = gdal.Open(file, gdal.GA_ReadOnly)

driver = gdal.GetDriverByName("GTiff")
options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']

# If not Multispectral, copy only the default odm_orthophoto
if not args.isMultispectral:
    typeName = 'RGB'

    if args.flir:
        typeName = 'Flir'

    if exifData['EXIF:SamplesPerPixel'] == 1:
        # Read bands
        gray = ds.GetRasterBand(1).ReadAsArray()
        if np.any(gray > 1):
            gray = gray / float(65535)

        gray = (gray * 255).astype(int)

        doGrayMap(typeName, gray)
    else:
        outFileNameRGB = os.path.join(outdir, args.outfile + typeName + ".tif")
        os.system("mv " + file + " " + outFileNameRGB)

        imgRGB = plt.imread(outFileNameRGB)
        cv2.imwrite(os.path.join(outdir, args.outfile + typeName + ".png"), imgRGB[:,:(2,1,0,3)], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        

else:

    # Read bands
    red =   ds.GetRasterBand(1).ReadAsArray()
    green = ds.GetRasterBand(2).ReadAsArray()
    blue =  ds.GetRasterBand(3).ReadAsArray()
    nir =   ds.GetRasterBand(4).ReadAsArray()
    redEdge = ds.GetRasterBand(5).ReadAsArray()
    thermal = None
    alpha = None

    if exifData['EXIF:SamplesPerPixel'] == 7:
        thermal = ds.GetRasterBand(6).ReadAsArray()
        alpha = ds.GetRasterBand(7).ReadAsArray()
    else:
        alpha = ds.GetRasterBand(6).ReadAsArray()

    if np.any(blue > 1) or np.any(green > 1) or np.any(red > 1) or np.any(nir > 1) or np.any(redEdge > 1):
        blue    = blue/float(65535)
        green   = green/float(65535)
        red     = red/float(65535)
        nir     = nir/float(65535)
        redEdge = redEdge/float(65535)
        if thermal is not None:
            thermal = thermal / float(65535)

    blue  = (blue*255).astype(int)
    green = (green*255).astype(int)
    red   = (red*255).astype(int)
    nir   = (nir*255).astype(int)
    redEdge = (redEdge*255).astype(int)
    if thermal is not None:
        thermal = (thermal*255).astype(int)
    alpha = (alpha*255).astype(int)

    # 
    alpha[(blue + green + red + nir + redEdge)!=0] = 255
    mask_alpha = (blue + green + red + nir + redEdge)==0

    [cols, rows] = blue.shape

    ## Export THERMAL
    if args.thermal:
        if thermal is not None:
            doGrayMap("THERMAL", thermal, alpha)

    ## Export RGB
    if args.rgb:
        outFileNameRGB = os.path.join(outdir, args.outfile + "RGBNoGamma.tif")
        outdata = driver.Create(outFileNameRGB, rows, cols, 4, gdal.GDT_Byte, options=options)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        # outdata.GetRasterBand(1).WriteArray(red  /float(65535)*255)
        # outdata.GetRasterBand(2).WriteArray(green/float(65535)*255)
        # outdata.GetRasterBand(3).WriteArray(blue /float(65535)*255)
        # outdata.GetRasterBand(4).WriteArray(alpha/float(65535)*255)
        outdata.GetRasterBand(1).WriteArray(red)
        outdata.GetRasterBand(2).WriteArray(green)
        outdata.GetRasterBand(3).WriteArray(blue)
        outdata.GetRasterBand(4).WriteArray(alpha)
        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        imgRGB = cv2.imread(outFileNameRGB, cv2.IMREAD_UNCHANGED)

        cv2.imwrite(os.path.join(outdir, args.outfile + "RGBNoGamma.png"), imgRGB, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        ## Gamma correction
        outFileNameRGBGamma = os.path.join(outdir, args.outfile + "RGB.tif")
        # Create an enhanced version of the RGB render using an unsharp mask
        gaussian_rgb = cv2.GaussianBlur(imgRGB[:,:,0:3]/255.0, (9,9), 10.0)
        gaussian_rgb[gaussian_rgb<0] = 0
        gaussian_rgb[gaussian_rgb>1] = 1
        unsharp_rgb = cv2.addWeighted(imgRGB[:,:,0:3]/255.0, 3, gaussian_rgb, -0.5, 0)
        unsharp_rgb[unsharp_rgb<0] = 0
        unsharp_rgb[unsharp_rgb>1] = 1

        # Apply a gamma correction to make the render appear closer to what our eyes would see
        gamma = args.gamma
        gamma_corr_rgb = unsharp_rgb**(1.0/gamma)

        gamma_corr_rgb *= 255
        gamma_corr_rgb = np.dstack((gamma_corr_rgb, alpha))

        outdata = driver.Create(outFileNameRGBGamma, rows, cols, 4, gdal.GDT_Byte, options=options)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        outdata.GetRasterBand(1).WriteArray(gamma_corr_rgb[:,:,2])
        outdata.GetRasterBand(2).WriteArray(gamma_corr_rgb[:,:,1])
        outdata.GetRasterBand(3).WriteArray(gamma_corr_rgb[:,:,0])
        outdata.GetRasterBand(4).WriteArray(alpha)
        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        cv2.imwrite(os.path.join(outdir, args.outfile + "RGB.png"), gamma_corr_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        resized = None
        gamma_corr_rgb = None

    ## Export NDVI
    if args.ndvi:
        outFileNameNDVI = os.path.join(outdir, args.outfile + "NDVI.tif")
        outdata = driver.Create(outFileNameNDVI, rows, cols, 4, gdal.GDT_Byte, options=options)

        # Make the values uint8
        # ndvi_red = red/float(65535)*255
        # ndvi_nir = nir/float(65535)*255
        ndvi_red = red.copy()
        ndvi_nir = nir.copy()

        # Calculate NDVI
        ndvi = np.zeros(ndvi_nir.shape, dtype=float)
        mask = np.not_equal((ndvi_nir + ndvi_red), 0.0)
        ndvi[mask] = np.true_divide(np.subtract(ndvi_nir[mask], ndvi_red[mask]), np.add(ndvi_nir[mask], ndvi_red[mask]))

        # Consider values between -0.4 and 1
        ndvi[mask] += 0.4
        ndvi[mask] /= (1+0.4)

        # Limit the values for further converting to uint8
        ndvi[ndvi>1] = 1
        ndvi[ndvi<0] = 0

        out_red   = np.zeros(red.shape)
        out_green = np.zeros(green.shape)
        out_blue  = np.zeros(blue.shape)
        out_alpha = np.zeros(alpha.shape)

        # Get the colormap values
        cmap = plt.get_cmap('RdYlGn')

        # Generate RGB image with the colormap values
        for i in range(cols):
            colors = cmap(ndvi[i,:])
            out_red[i,:]   = (colors[:,0]*255).astype(int)
            out_green[i,:] = (colors[:,1]*255).astype(int)
            out_blue[i,:]  = (colors[:,2]*255).astype(int)
            out_alpha[i,:] = (colors[:,3]*255).astype(int)

        out_alpha[mask_alpha] = 0

        # Export result
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        outdata.GetRasterBand(1).WriteArray(out_red)
        outdata.GetRasterBand(2).WriteArray(out_green)
        outdata.GetRasterBand(3).WriteArray(out_blue)
        outdata.GetRasterBand(4).WriteArray(out_alpha)

        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        imgNDVI = cv2.imread(outFileNameNDVI, cv2.IMREAD_UNCHANGED)

        cv2.imwrite(os.path.join(outdir, args.outfile + "NDVI.png"), imgNDVI, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    ## Export CIR
    if args.cir:
        outFileNameCIR = os.path.join(outdir, args.outfile + "CIR.tif")
        outdata = driver.Create(outFileNameCIR, rows, cols, 4, gdal.GDT_Byte, options=options)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        # outdata.GetRasterBand(1).WriteArray(nir  /float(65535)*255)
        # outdata.GetRasterBand(2).WriteArray(green/float(65535)*255)
        # outdata.GetRasterBand(3).WriteArray(red  /float(65535)*255)
        # outdata.GetRasterBand(4).WriteArray(alpha/float(65535)*255)
        outdata.GetRasterBand(1).WriteArray(nir)
        outdata.GetRasterBand(2).WriteArray(red)
        outdata.GetRasterBand(3).WriteArray(green)
        outdata.GetRasterBand(4).WriteArray(alpha)
        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        imgCIR = cv2.imread(outFileNameCIR, cv2.IMREAD_UNCHANGED)

        # Generate png images
        cv2.imwrite(os.path.join(outdir, args.outfile + "CIR.png"), imgCIR, [cv2.IMWRITE_PNG_COMPRESSION, 9])


    ## Export NDRE
    if args.ndre:
        outFileNameNDRE = os.path.join(outdir, args.outfile + "NDRE.tif")
        outdata = driver.Create(outFileNameNDRE, rows, cols, 4, gdal.GDT_Byte, options=options)

        # Make the values uint8
        ndre_redEdge = redEdge.copy()
        ndre_nir = nir.copy()

        # Calculate NDRE
        ndre = np.zeros(ndre_nir.shape, dtype=float)
        mask = np.not_equal((ndre_nir + ndre_redEdge), 0.0)
        ndre[mask] = np.true_divide(np.subtract(ndre_nir[mask], ndre_redEdge[mask]), np.add(ndre_nir[mask], ndre_redEdge[mask]))

        # Consider values between -1 and 1
        ndre[mask] += 1
        ndre[mask] /= (1+1)

        # Limit the values for further converting to uint8
        ndre[ndre>1] = 1
        ndre[ndre<0] = 0

        out_red = np.zeros(red.shape)
        out_green = np.zeros(green.shape)
        out_blue = np.zeros(blue.shape)
        out_alpha = np.zeros(alpha.shape)

        # Get the colormap values
        cmap = plt.get_cmap('Spectral')

        # Generate RGB image with the colormap values
        for i in range(cols):
            colors = cmap(ndre[i,:])
            out_red[i,:]   = (colors[:,0]*255).astype(int)
            out_green[i,:] = (colors[:,1]*255).astype(int)
            out_blue[i,:]  = (colors[:,2]*255).astype(int)
            out_alpha[i,:] = (colors[:,3]*255).astype(int)

        out_alpha[mask_alpha] = 0

        # Export result
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        outdata.GetRasterBand(1).WriteArray(out_red)
        outdata.GetRasterBand(2).WriteArray(out_green)
        outdata.GetRasterBand(3).WriteArray(out_blue)
        outdata.GetRasterBand(4).WriteArray(out_alpha)

        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        imgNDRE = cv2.imread(outFileNameNDRE, cv2.IMREAD_UNCHANGED)

        # Generate png images
        cv2.imwrite(os.path.join(outdir, args.outfile + "NDRE.png"), imgNDRE, [cv2.IMWRITE_PNG_COMPRESSION, 9])


    ## Export NDWI
    if args.ndwi:
        outFileNameNDWI = os.path.join(outdir, args.outfile + "NDWI.tif")
        outdata = driver.Create(outFileNameNDWI, rows, cols, 4, gdal.GDT_Byte, options=options)

        # Make the values uint8
        ndwi_green = green.copy()
        ndwi_nir = nir.copy()

        # Calculate NDWI
        ndwi = np.zeros(ndwi_nir.shape, dtype=float)
        mask = np.not_equal((ndwi_nir + ndwi_green), 0.0)
        ndwi[mask] = np.true_divide(np.subtract(ndwi_green[mask], ndwi_nir[mask]), np.add(ndwi_green[mask], ndwi_nir[mask]))

        # Consider values between -1 and 1
        ndwi[mask] += 1
        ndwi[mask] /= (1+1)

        # Limit the values for further converting to uint8
        ndwi[ndwi>1] = 1
        ndwi[ndwi<0] = 0

        out_red   = np.zeros(red.shape)
        out_green = np.zeros(green.shape)
        out_blue  = np.zeros(blue.shape)
        out_alpha = np.zeros(alpha.shape)

        # Get the colormap values
        cmap = plt.get_cmap('Spectral')

        # Generate RGB image with the colormap values
        for i in range(cols):
            colors = cmap(ndwi[i,:])
            out_red[i,:]   = (colors[:,0]*255).astype(int)
            out_green[i,:] = (colors[:,1]*255).astype(int)
            out_blue[i,:]  = (colors[:,2]*255).astype(int)
            out_alpha[i,:] = (colors[:,3]*255).astype(int)

        out_alpha[mask_alpha] = 0

        # Export result
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input

        outdata.GetRasterBand(1).WriteArray(out_red)
        outdata.GetRasterBand(2).WriteArray(out_green)
        outdata.GetRasterBand(3).WriteArray(out_blue)
        outdata.GetRasterBand(4).WriteArray(out_alpha)

        outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
        outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
        outdata.FlushCache() ##saves to disk!!

        outdata = None

        imgNDWI = cv2.imread(outFileNameNDWI, cv2.IMREAD_UNCHANGED)

        # Generate png images
        cv2.imwrite(os.path.join(outdir, args.outfile + "NDWI.png"), imgNDWI, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    ds=None
