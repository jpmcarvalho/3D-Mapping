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

def deignThermal(inputFile, outPutFile):
    imageNames = inputFile
    # print(imageNames)

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
    band_filename = inputFile
    filename = outPutFile
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

