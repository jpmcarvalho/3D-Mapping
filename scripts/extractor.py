#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import flirimageextractor
from matplotlib import cm
import cv2 as cv2
import os, glob
import argparse
import shutil

# Argument parser
parser = argparse.ArgumentParser(description='Correct Micasense reflectance images. Update EXIF information.')
#  parser.add_argument("-b", "--base", type=str, default='/Users/jpc/Documents/PDMFC/data', help='Base directory')
#/code/data
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='flir_output', help='File directory')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
parser.add_argument("--isMultispectral", help="Preprocess Multispectral images", action="store_true")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

inputDir = os.path.join(args.base, args.uid, 'data_raw/Mission')
outputDir = os.path.join(args.base, args.uid, args.dir)

if os.path.exists(outputDir):
    shutil.rmtree(outputDir)

os.mkdir(outputDir)

pw = os.getcwd()

for root, dirs, files in os.walk(os.path.join(inputDir)):
    files = sorted(files)
    while len(files):
        file = files[0]
        
        
        file_path = os.path.join(inputDir, file)
        
        
        flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
        flir.process_image(os.path.join(inputDir, file))
        imgThermal = flir.get_thermal_np()
        valor0 = 0  # -10 graus
        valor50 = 100  # 50 graus
        imgThermal[imgThermal < valor0] = valor0
        imgThermal[imgThermal > valor50] = valor50
        imgThermal = imgThermal - valor0
        imgThermal = (imgThermal / (valor50 - valor0))

        maxValue = (2 ** 8) - 1
        imgThermal *= maxValue



        cv2.imwrite(os.path.join(outputDir, file), imgThermal)

        #shFile = os.path.join(args.base, 'copyExif_JPEG.sh')
        shFile = 'copyExif_JPEG.sh'
        # Copy all valid EXIF information
        ptoPath = shFile + ' ' + os.path.join(inputDir, file) + ' ' + os.path.join(outputDir, file)
        print(ptoPath)
        os.system("bash " + ptoPath)

        del files[0]

    if os.path.exists(os.path.join(outputDir)):
        fileList = glob.glob(os.path.join(outputDir, "*_original"))

        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")
