#!/usr/bin/python3
import sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages/')

import argparse
import exiftool
import os
import cv2
import laspy
import numpy as np


user = os.environ.get('USER')

parser = argparse.ArgumentParser(description='Slicing a complete orthophoto into smaller images.')
parser.add_argument("-id", "--id", type=str, default="15", help='User ID')
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-f", "--file", type=str, default='/code/data/15/maps/mapRGB.tif', help='Tif file path')
parser.add_argument("-p", "--pcd", type=str, default='/code/data/15/odm_georeferencing/odm_georeferenced_model.laz', help='Laz file path')
parser.add_argument(      "--ndvi", action="store_true", help='NDVI.')
parser.add_argument(      "--rgb",  action="store_true", help='RGB.')
parser.add_argument(      "--rgbng",action="store_true", help='RGBNoGamma.')
parser.add_argument(      "--ndre", action="store_true", help='NDRE.')
parser.add_argument(      "--cir",  action="store_true", help='CIR.')
parser.add_argument(      "--ndwi", action="store_true", help='NDWI.')
parser.add_argument(      "--thermal", action="store_true", help='Thermal.')
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

images_path = []

if args.id:
	file_path = os.path.join(args.base, args.id, 'maps/map')

	if args.ndvi:
		images_path.append('NDVI')

	if args.rgb:
		images_path.append('RGB')

	#elif args.rgbng:
	#	images_path.append('RGBNoGamma.tif'

	if args.ndre:
		images_path.append('NDRE')

	if args.cir:
		images_path.append('CIR')

	if args.ndwi:
		images_path.append('NDWI')

	if args.thermal:
		images_path.append('THERMAL')

	pcd_path = os.path.join(args.base, args.id,'odm_georeferencing/odm_georeferenced_model.laz')
else:
	file_path = args.file
	pcd_path = args.pcd

if not os.path.exists(pcd_path):
		print("File "+pcd_path+" does not exist.")
		exit()

cut_name, extension = os.path.splitext(pcd_path)

for index in images_path:

    img_path = file_path + index + '.tif'
    if not os.path.exists(img_path):
        print("File " + img_path + " does not exist.")
        exit()

    print("Reading EXIF info...")

    with exiftool.ExifTool() as et:
        pixelScale = et.get_tag("PixelScale", img_path)
        tiePoint = et.get_tag("EXIF:ModelTiePoint", img_path)

    pixelScale = [float(i) for i in pixelScale.split()]
    tiePoint = [float(i) for i in tiePoint.split()]

    print("Reading image file...")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    print("Reading PCD file with laspy...")
    las = laspy.read(pcd_path)

    las_scale = las.header.scales
    las_offset = las.header.offsets

    print("Processing...")

    # Ajuste dos tie points com offsets
    tiePoint[3:5] = np.subtract(tiePoint[3:5], las_offset[:2])

    data_values = np.vstack((las.X, las.Y))
    pixel = np.vstack((
        (tiePoint[4] - data_values[1, :] * las_scale[1]) / pixelScale[1],
        (data_values[0, :] * las_scale[0] - tiePoint[3]) / pixelScale[0]
    )).astype(int)

    pixel[0][pixel[0, :] >= img.shape[0]] = img.shape[0] - 1
    pixel[1][pixel[1, :] >= img.shape[1]] = img.shape[1] - 1

    print("Mapping RGB from image...")

    blue, green, red = np.transpose(img[pixel[0], pixel[1], 0:3])

    print("Creating new LAS file with RGB...")

    las.red = red.astype(np.uint16) * 256
    las.green = green.astype(np.uint16) * 256
    las.blue = blue.astype(np.uint16) * 256

    out_path = cut_name + index + 'colorFixed.laz'
    las.write(out_path)

    print(f"Wrote: {out_path}")