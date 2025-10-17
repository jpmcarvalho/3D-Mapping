#!/usr/bin/env python

import numpy as np
import sys

sys.path.insert(1, '/usr/local/lib/python3.5/dist-packages/')
import cv2 as cv2
import matplotlib.pyplot as plt
import argparse
import os
from osgeo import gdal

# Argument parser
parser = argparse.ArgumentParser(description='Generate NDVI from odm_orthophoto.')
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='odm_orthophoto', help='File directory')
parser.add_argument("-f", "--file", type=str, default='odm_orthophoto.tif', help='Input file.')
parser.add_argument("-O", "--outdir", type=str, default='maps', help='Output directory')
parser.add_argument("-o", "--outfile", type=str, default='map', help='Output file.')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
parser.add_argument("-g", "--gamma", type=float, default='1.2', help='Gamma Correction factor')
parser.add_argument("--ndvi", action="store_true", help='NDVI.')
parser.add_argument("--rgb", action="store_true", help='RGB.')
parser.add_argument("--ndre", action="store_true", help='NDRE.')
parser.add_argument("--cir", action="store_true", help='CIR.')
parser.add_argument("--ndwi", action="store_true", help='NDWI.')
parser.add_argument("--isMultispectral", help="Post-process Multispectral images", action="store_true")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()


def homograpyMaps(img1, img2, imgToTransform):
    features = cv2.xfeatures2d.SIFT_create()
    # features = cv2.xfeatures2d.SURF_create()
    # features = cv2.ORB_create(nfeatures=1500)

    keypoints_left, descriptors_left = features.detectAndCompute(img1, None)
    keypoints_right, descriptors_right = features.detectAndCompute(img2, None)

    # img1 = cv2.drawKeypoints(img1, keypoints_left, None)
    # img2 = cv2.drawKeypoints(img2, keypoints_left, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_left, descriptors_right, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # -- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, keypoints_left, img2, keypoints_right, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    rows, cols = img1.shape
    for i in range(len(good_matches)):
        diffy = abs(keypoints_left[good_matches[i].queryIdx].pt[1] - keypoints_right[good_matches[i].trainIdx].pt[1])
        diffx = abs(keypoints_left[good_matches[i].queryIdx].pt[0] - keypoints_right[good_matches[i].trainIdx].pt[0])

        g = (diffy / rows) * 100.0
        h = (diffx / cols) * 100.0

        if g < 10 and h < 10:
            # -- Get the keypoints from the good matches
            obj[i, 0] = keypoints_left[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_left[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_right[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_right[good_matches[i].trainIdx].pt[1]

    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
    # -- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = img1.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = img1.shape[1]
    obj_corners[2, 0, 1] = img1.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = img1.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, H)
    # # -- Draw lines between the corners (the mapped object in the scene - image_2 )
    # cv2.line(img_matches, (int(scene_corners[0, 0, 0] + img1.shape[1]), int(scene_corners[0, 0, 1])),
    #          (int(scene_corners[1, 0, 0] + img1.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
    # cv2.line(img_matches, (int(scene_corners[1, 0, 0] + img1.shape[1]), int(scene_corners[1, 0, 1])),
    #          (int(scene_corners[2, 0, 0] + img1.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
    # cv2.line(img_matches, (int(scene_corners[2, 0, 0] + img1.shape[1]), int(scene_corners[2, 0, 1])),
    #          (int(scene_corners[3, 0, 0] + img1.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
    # cv2.line(img_matches, (int(scene_corners[3, 0, 0] + img1.shape[1]), int(scene_corners[3, 0, 1])),
    #          (int(scene_corners[0, 0, 0] + img1.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)
    # # -- Show detected matches
    # img_matches = cv2.resize(img_matches, (int(1260), int(680)))
    #
    # cv2.imshow('taste', img_matches)
    # cv2.waitKey(0)

    return cv2.warpPerspective(imgToTransform, H, (img1.shape[1], img1.shape[0]), cv2.INTER_LINEAR, cv2.BORDER_REFLECT)


def calcNDVI(redAlignment, nirAlignment):
    ndvi_red = np.copy(redAlignment).astype(float)
    ndvi_nir = np.copy(nirAlignment).astype(float)

    # Calculate NDVI
    ndvi = np.zeros(ndvi_nir.shape, dtype=float)
    mask = np.not_equal((ndvi_nir + ndvi_red), 0.0)
    ndvi[mask] = np.true_divide(np.subtract(ndvi_nir[mask], ndvi_red[mask]), np.add(ndvi_nir[mask], ndvi_red[mask]))

    # Consider values between -0.4 and 1
    ndvi[mask] += 0.2
    ndvi[mask] /= (1 + 0.2)
    # ndvi = np.ma.masked_where(nirAlignment < 40, ndvi)

    # Limit the values for further converting to uint8
    ndvi[ndvi > 1] = 1
    ndvi[ndvi < 0] = 0
    ndvi[0, 0] = 0
    ndvi[0, 1] = 1

    out_ndvi = np.zeros((nirAlignment.shape[0], nirAlignment.shape[1], 4), np.uint8)

    # Get the colormap values
    cmap = plt.get_cmap('RdYlGn')

    out_red = np.zeros(redAlignment.shape)
    out_green = np.zeros(redAlignment.shape)
    out_blue = np.zeros(redAlignment.shape)

    [cols, _] = out_blue.shape
    # Generate RGB image with the colormap values
    for i in range(cols):
        colors = cmap(ndvi[i, :])
        out_blue[i, :] = (colors[:, 2] * 255).astype(int)
        out_green[i, :] = (colors[:, 1] * 255).astype(int)
        out_red[i, :] = (colors[:, 0] * 255).astype(int)

    out_ndvi[:, :, 0] = (out_blue[:, :]).astype(int)
    out_ndvi[:, :, 1] = (out_green[:, :]).astype(int)
    out_ndvi[:, :, 2] = (out_red[:, :]).astype(int)
    out_ndvi[:, :, 3] = alpha[:, :].astype(int)

    # ndvibw = np.zeros((nirAlignment.shape[0], nirAlignment.shape[1]), np.uint8)
    # ndvibw[:, :] = (ndvi[:, :] * 255).astype(int)
    # cv2.imwrite('/home/jmcarvalho/Desktop/ndvibw.png', ndvibw)

    return out_ndvi


def calcNDRE(redEdgeAlignment, nirAlignment):
    ndre_redEdge = np.copy(redEdgeAlignment).astype(float)
    ndre_nir = np.copy(nirAlignment).astype(float)

    # Calculate NDRE
    ndre = np.zeros(ndre_redEdge.shape, dtype=float)
    mask = np.not_equal((ndre_nir + ndre_redEdge), 0.0)
    ndre[mask] = np.true_divide(np.subtract(ndre_nir[mask], ndre_redEdge[mask]),
                                np.add(ndre_nir[mask], ndre_redEdge[mask]))

    # Consider values between -1 and 1
    ndre[mask] += 1
    ndre[mask] /= (1 + 1)

    # Limit the values for further converting to uint8
    ndre[ndre > 1] = 1
    ndre[ndre < 0] = 0

    out_ndre = np.zeros((redEdgeAlignment.shape[0], redEdgeAlignment.shape[1], 4), np.uint8)

    # Get the colormap values
    cmap = plt.get_cmap('Spectral')

    out_red = np.zeros(redEdgeAlignment.shape)
    out_green = np.zeros(redEdgeAlignment.shape)
    out_blue = np.zeros(redEdgeAlignment.shape)

    [cols, _] = out_blue.shape
    # Generate RGB image with the colormap values
    for i in range(cols):
        colors = cmap(ndre[i, :])
        out_blue[i, :] = (colors[:, 2] * 255).astype(int)
        out_green[i, :] = (colors[:, 1] * 255).astype(int)
        out_red[i, :] = (colors[:, 0] * 255).astype(int)

    out_ndre[:, :, 0] = (out_blue[:, :]).astype(int)
    out_ndre[:, :, 1] = (out_green[:, :]).astype(int)
    out_ndre[:, :, 2] = (out_red[:, :]).astype(int)
    out_ndre[:, :, 3] = alpha[:, :].astype(int)

    return out_ndre


def calcNDWI(greenAlignment, nirAlignment):
    ndwi_green = np.copy(greenAlignment).astype(float)
    ndwi_nir = np.copy(nirAlignment).astype(float)

    # Calculate NDWI
    ndwi = np.zeros(ndwi_nir.shape, dtype=float)
    mask = np.not_equal((ndwi_nir + ndwi_green), 0.0)
    ndwi[mask] = np.true_divide(np.subtract(ndwi_green[mask], ndwi_nir[mask]), np.add(ndwi_green[mask], ndwi_nir[mask]))

    # Consider values between -1 and 1
    ndwi[mask] += 1
    ndwi[mask] /= (1 + 1)

    # Limit the values for further converting to uint8
    ndwi[ndwi > 1] = 1
    ndwi[ndwi < 0] = 0

    out_ndwi = np.zeros((greenAlignment.shape[0], greenAlignment.shape[1], 4), np.uint8)

    # Get the colormap values
    cmap = plt.get_cmap('Spectral')

    out_red = np.zeros(greenAlignment.shape)
    out_green = np.zeros(greenAlignment.shape)
    out_blue = np.zeros(greenAlignment.shape)

    [cols, _] = out_blue.shape
    # Generate RGB image with the colormap values
    for i in range(cols):
        colors = cmap(ndwi[i, :])
        out_blue[i, :] = (colors[:, 2] * 255).astype(int)
        out_green[i, :] = (colors[:, 1] * 255).astype(int)
        out_red[i, :] = (colors[:, 0] * 255).astype(int)

    out_ndwi[:, :, 0] = (out_blue[:, :]).astype(int)
    out_ndwi[:, :, 1] = (out_green[:, :]).astype(int)
    out_ndwi[:, :, 2] = (out_red[:, :]).astype(int)
    out_ndwi[:, :, 3] = alpha[:, :].astype(int)

    return out_ndwi


def gammaCorrection(img, weigth=5, gamma=1.2, beta=-0.5):
    if len(img.shape) == 3:
        gaussian = cv2.GaussianBlur(img[:, :, 0:3] / 255.0, (9, 9), 10.0)
    else:
        gaussian = cv2.GaussianBlur(img[:, :] / 255.0, (9, 9), 10.0)
    gaussian[gaussian < 0] = 0
    gaussian[gaussian > 1] = 1
    if len(img.shape) == 3:
        unsharp = cv2.addWeighted(img[:, :, 0:3] / 255.0, weigth, gaussian, beta, 0)
    else:
        unsharp = cv2.addWeighted(img[:, :] / 255.0, weigth, gaussian, beta, 0)
    unsharp[unsharp < 0] = 0
    unsharp[unsharp > 1] = 1
    gamma_corr = unsharp ** (1.0 / gamma)
    if len(img.shape) == 3:
        gammaFinal = np.zeros((img.shape[0], img.shape[1], 4), np.uint8)
        gammaFinal[:, :, 0:3] = (gamma_corr[:, :, 0:3] * 255.0).astype(int)
        gammaFinal[:, :, 3] = img[:, :, 3]
    else:
        gammaFinal = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        gammaFinal[:, :] = (gamma_corr[:, :] * 255.0).astype(int)
    return gammaFinal


# Generate input file and output directory path strings
fileBlue = os.path.join(args.base, args.uid, args.dir + '_1', args.file)
fileGreen = os.path.join(args.base, args.uid, args.dir + '_2', args.file)
fileRed = os.path.join(args.base, args.uid, args.dir + '_3', args.file)
fileNir = os.path.join(args.base, args.uid, args.dir + '_4', args.file)
fileRedEdge = os.path.join(args.base, args.uid, args.dir + '_5', args.file)
outdir = os.path.join(args.base, args.uid, args.outdir)

if args.isMultispectral:
    if not os.path.exists(fileBlue) or not os.path.exists(fileGreen) or not os.path.exists(fileRed) or not os.path.exists(
            fileNir) or not os.path.exists(fileRedEdge):
        print("File not found!")
        exit()

os.makedirs(outdir, 0o777, True)

# If not Multispectral, copy only the default odm_orthophoto
if not args.isMultispectral:
    outFileNameRGB = os.path.join(outdir, args.outfile + "RGB.tif")
    os.system("mv " + fileBlue + " " + outFileNameRGB)

    imgRGB = cv2.imread(outFileNameRGB, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(os.path.join(outdir, args.outfile + "RGB.png"), imgRGB)
    exit()

# Open Multispectral tif
dsBlue = gdal.Open(fileBlue, gdal.GA_ReadOnly)
dsGreen = gdal.Open(fileGreen, gdal.GA_ReadOnly)
dsRed = gdal.Open(fileRed, gdal.GA_ReadOnly)
dsNir = gdal.Open(fileNir, gdal.GA_ReadOnly)
dsRedEdge = gdal.Open(fileRedEdge, gdal.GA_ReadOnly)

driver = gdal.GetDriverByName("GTiff")
options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']

# Read bands
blue = dsBlue.GetRasterBand(1).ReadAsArray()
green = dsGreen.GetRasterBand(1).ReadAsArray()
red = dsRed.GetRasterBand(1).ReadAsArray()
nir = dsNir.GetRasterBand(1).ReadAsArray()
redEdge = dsRedEdge.GetRasterBand(1).ReadAsArray()

if np.any(blue > 1) or np.any(green > 1) or np.any(red > 1) or np.any(nir > 1) or np.any(redEdge > 1):
    blue = blue / float(65535)
    green = green / float(65535)
    red = red / float(65535)
    nir = nir / float(65535)
    redEdge = redEdge / float(65535)

blue = (blue * 255).astype(np.uint8)
green = (green * 255).astype(np.uint8)
red = (red * 255).astype(np.uint8)
nir = (nir * 255).astype(np.uint8)
redEdge = (redEdge * 255).astype(np.uint8)

blue = cv2.resize(blue, (redEdge.shape[1], redEdge.shape[0]))
green = cv2.resize(green, (redEdge.shape[1], redEdge.shape[0]))
red = cv2.resize(red, (redEdge.shape[1], redEdge.shape[0]))
nir = cv2.resize(nir, (redEdge.shape[1], redEdge.shape[0]))
# redEdge = cv2.resize(redEdge, (nir.shape[0], nir.shape[1]))

gammaBlue = gammaCorrection(blue)
gammaGreen = gammaCorrection(green)
gammaRed = gammaCorrection(red)
gammaNir = gammaCorrection(nir, 2)
gammaRedEdge = gammaCorrection(redEdge, 2)

blueAlignment = homograpyMaps(gammaBlue, gammaRedEdge, blue)
greenAlignment = homograpyMaps(gammaGreen, gammaRedEdge, green)
redAlignment = homograpyMaps(gammaRed, gammaRedEdge, red)
nirAlignment = homograpyMaps(gammaNir, gammaRedEdge, nir)

blueAlignment = cv2.resize(blueAlignment, (redEdge.shape[1], redEdge.shape[0]))
greenAlignment = cv2.resize(greenAlignment, (redEdge.shape[1], redEdge.shape[0]))
redAlignment = cv2.resize(redAlignment, (redEdge.shape[1], redEdge.shape[0]))
nirAlignment = cv2.resize(nirAlignment, (redEdge.shape[1], redEdge.shape[0]))
redEdgeAlignment = cv2.resize(redEdge, (redEdge.shape[1], redEdge.shape[0]))

alpha = np.zeros((redEdge.shape[0], redEdge.shape[1]), np.uint8)
alpha[(blueAlignment + greenAlignment + redAlignment + nirAlignment + redEdgeAlignment) != 0] = 255
mask_alpha = (blueAlignment + greenAlignment + redAlignment + nirAlignment + redEdgeAlignment) == 0

## Export RGB
if args.rgb:
    outFileNameRGB = os.path.join(outdir, args.outfile + "RGBNoGamma.tif")
    outdata = driver.Create(outFileNameRGB, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(redAlignment)
    outdata.GetRasterBand(2).WriteArray(greenAlignment)
    outdata.GetRasterBand(3).WriteArray(blueAlignment)
    outdata.GetRasterBand(4).WriteArray(alpha)
    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    imgRGB = cv2.imread(outFileNameRGB, cv2.IMREAD_UNCHANGED)

    cv2.imwrite(os.path.join(outdir, args.outfile + "RGBNoGamma.png"), imgRGB)

    ## Gamma correction
    outFileNameRGBGamma = os.path.join(outdir, args.outfile + "RGB.tif")

    rgb = np.zeros((redEdgeAlignment.shape[0], redEdgeAlignment.shape[1], 4), np.uint8)
    rgb[:, :, 0] = blueAlignment[:, :]
    rgb[:, :, 1] = greenAlignment[:, :]
    rgb[:, :, 2] = redAlignment[:, :]
    rgb[:, :, 3] = alpha[:, :]

    rgb = gammaCorrection(rgb, 3)

    outdata = driver.Create(outFileNameRGBGamma, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(rgb[:, :, 2])
    outdata.GetRasterBand(2).WriteArray(rgb[:, :, 1])
    outdata.GetRasterBand(3).WriteArray(rgb[:, :, 0])
    outdata.GetRasterBand(4).WriteArray(alpha)
    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    cv2.imwrite(os.path.join(outdir, args.outfile + "RGB.png"), rgb)

## Export CIR
if args.cir:
    outFileNameCIR = os.path.join(outdir, args.outfile + "CIR.tif")
    outdata = driver.Create(outFileNameCIR, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(nirAlignment)
    outdata.GetRasterBand(2).WriteArray(redAlignment)
    outdata.GetRasterBand(3).WriteArray(greenAlignment)
    outdata.GetRasterBand(4).WriteArray(alpha)
    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    imgCIR = cv2.imread(outFileNameCIR, cv2.IMREAD_UNCHANGED)

    # Generate png images
    cv2.imwrite(os.path.join(outdir, args.outfile + "CIR.png"), imgCIR)

## Export NDVI
if args.ndvi:
    outFileNameNDVI = os.path.join(outdir, args.outfile + "NDVI.tif")
    outdata = driver.Create(outFileNameNDVI, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)

    out_ndvi = calcNDVI(redAlignment, nirAlignment)
    # Export result
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(out_ndvi[:, :, 2])
    outdata.GetRasterBand(2).WriteArray(out_ndvi[:, :, 1])
    outdata.GetRasterBand(3).WriteArray(out_ndvi[:, :, 0])
    outdata.GetRasterBand(4).WriteArray(out_ndvi[:, :, 3])

    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    imgNDVI = cv2.imread(outFileNameNDVI, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(outdir, args.outfile + "NDVI.png"), imgNDVI)

## Export NDRE
if args.ndre:
    outFileNameNDRE = os.path.join(outdir, args.outfile + "NDRE.tif")
    outdata = driver.Create(outFileNameNDRE, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)

    out_ndre = calcNDRE(redEdgeAlignment, nirAlignment)
    # Export result
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(out_ndre[:, :, 2])
    outdata.GetRasterBand(2).WriteArray(out_ndre[:, :, 1])
    outdata.GetRasterBand(3).WriteArray(out_ndre[:, :, 0])
    outdata.GetRasterBand(4).WriteArray(out_ndre[:, :, 3])

    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None

    imgNDRE = cv2.imread(outFileNameNDRE, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(outdir, args.outfile + "NDRE.png"), imgNDRE)

## Export NDWI
if args.ndwi:
    outFileNameNDWI = os.path.join(outdir, args.outfile + "NDWI.tif")
    outdata = driver.Create(outFileNameNDWI, redEdgeAlignment.shape[1], redEdgeAlignment.shape[0], 4, gdal.GDT_Byte,
                            options=options)

    out_ndwi = calcNDWI(greenAlignment, nirAlignment)
    # Export result
    outdata.SetGeoTransform(dsRedEdge.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(dsRedEdge.GetProjection())  ##sets same projection as input

    outdata.GetRasterBand(1).WriteArray(out_ndwi[:, :, 2])
    outdata.GetRasterBand(2).WriteArray(out_ndwi[:, :, 1])
    outdata.GetRasterBand(3).WriteArray(out_ndwi[:, :, 0])
    outdata.GetRasterBand(4).WriteArray(out_ndwi[:, :, 3])

    outdata.GetRasterBand(1).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(2).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(3).SetNoDataValue(255)  ##if you want these values transparent
    outdata.GetRasterBand(4).SetNoDataValue(0)  ##if you want these values transparent
    outdata.FlushCache()  ##saves to disk!!

    outdata = None
    imgNDWI = cv2.imread(outFileNameNDWI, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(outdir, args.outfile + "NDWI.png"), imgNDWI)

# cv2.imwrite('/home/jmcarvalho/Desktop/blueAlignment.png', gammaBlue)
# cv2.imwrite('/home/jmcarvalho/Desktop/greenAlignment.png', gammaGreen)
# cv2.imwrite('/home/jmcarvalho/Desktop/redAlignment.png', gammaRed)
# cv2.imwrite('/home/jmcarvalho/Desktop/nirAlignment.png', gammaNir)
# cv2.imwrite('/home/jmcarvalho/Desktop/redEdgeAlignment.png', gammaRedEdge)
cv2.destroyAllWindows()
