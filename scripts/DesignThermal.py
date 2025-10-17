#!/usr/bin/python

import cv2 as cv2
import numpy as np
import imageio
import argparse
import exiftool
import os, glob
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils

import micasense.metadata as metadata
from datetime import datetime, timedelta

from os import path

import imageio

# Argument parser
parser = argparse.ArgumentParser(description='Correct Micasense reflectance images. Update EXIF information.')
parser.add_argument("-b", "--base", type=str, default='/home/draco/Documents/colorThermal', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='data_raw', help='File directory')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
parser.add_argument("-o", "--outdir", type=str, default='images', help='Output file directory.')
parser.add_argument("-s", "--splitImages", type=int, default=150, help='Maximum number of images per group.')
parser.add_argument("--isMultispectral", help="Preprocess Multispectral images", action="store_true")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

if not args.dir:
    print("Please specify the path to the images.")
    exit()

if not args.outdir:
    outdir = os.getcwd()
else:
    outdir = os.path.join(args.base, args.uid, args.outdir)

# Create needed directories to store the generated images
os.makedirs(outdir, 0o777, True)

imagePath = os.path.join(args.base, args.uid, args.dir)

# Get Mission files
for root, dirs, files in os.walk(os.path.join(imagePath, 'Mission')):
    numberFiles = int(len(files))
    filesThermal = sorted(filter(lambda f: f.endswith("_6.tif"), files))

first_file = filesThermal[0]

print("First file:", first_file)

# While there is images to treat
index = 0
while (index < len(filesThermal)):

    # Global path
    imageNames = os.path.join(imagePath, 'Mission', filesThermal[index])
    print(imageNames)

    imgThermal = cv2.imread(os.path.join(imageNames), -1)
    # imgBlue    = cv2.imread('IMG_0000_1.tif', -1)
    # width, height = imgBlue.shape

    valor0 = 26315.0  # -10 graus
    valor50 = 32315.0  # 50 graus
    imgThermal[imgThermal < valor0] = valor0
    imgThermal[imgThermal > valor50] = valor50
    imgThermal = imgThermal - valor0
    maxValue = (2 ** 16) - 1
    imgThermal = (imgThermal / (valor50 - valor0))

    imgThermal *= maxValue

    imgThermal = cv2.resize(imgThermal, (2064, 1544))

    # thermalPercentage = 0.7
    # imgCombinedBlueThermal = imgBlue*(1-thermalPercentage) + imgThermal*thermalPercentage

    # imgThermal1 = np.copy(255.0*imgThermal/maxValue).astype(np.uint8)
    # imgThermalColor = cv2.applyColorMap(imgThermal1, cv2.COLORMAP_JET)
    # imgThermalColor = cv2.resize(imgThermalColor, (640, 480))
    # cv2.imshow('tt', imgThermalColor)
    # cv2.waitKey(0)

    # imageio.imwrite('result.tif', imgThermal.astype('uint16'))

    # Get original image metadata
    md = metadata.Metadata(imageNames)

    # Get second deviation
    subsec = int(md.get_item('EXIF:SubSecTime'))
    negative = False

    # Get Time of Creation
    name = md.get_item('EXIF:DateTimeOriginal')
    utc_time = datetime.strptime(name, "%Y:%m:%d %H:%M:%S")

    # If the value is negative, exiftool will ignore it
    if subsec < 0:
        negative = True
        subsec *= -1.0
        subsec = float('0.{}'.format(int(subsec)))
        subsec = 1 - subsec
        subsec *= 1e5
        subsec = int(subsec)

        utc_time -= timedelta(seconds=1)

    # Update EXIF parameters
    md.exif['EXIF:SubSecTime'] = subsec
    md.exif['EXIF:DateTimeOriginal'] = utc_time
    md.exif['EXIF:CreateDate'] = utc_time

    # Save corrected image uint16 TIFF
    band_filename = os.path.join(imagePath, 'Mission', filesThermal[index])
    filename = os.path.join(outdir, filesThermal[index])
    imageio.imwrite(filename, imgThermal.astype('uint16'))

    # Copy all valid EXIF information
    ptoPath = 'copyExif.sh ' + band_filename + ' ' + filename
    os.system("bash " + ptoPath)

    if negative:
        # Write the updated EXIF values if subsec was updated
        with exiftool.ExifTool() as exift:
            str_out_subsec = '-EXIF:SubSecTime=' + str(md.exif['EXIF:SubSecTime']).zfill(5)
            str_out_date = '-EXIF:DateTimeOriginal=' + str(md.exif['EXIF:DateTimeOriginal'])
            str_out_createdate = '-EXIF:CreateDate=' + str(md.exif['EXIF:CreateDate'])

            exift.execute(str_out_subsec.encode(), str_out_date.encode(), str_out_createdate.encode(),
                          filename.encode())

    # Remove images from list
    index += 1

    if os.path.exists(os.path.join(outdir)):
        fileList = glob.glob(os.path.join(outdir, "*_original"))
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except OSError:
            print("Error while deleting file")
