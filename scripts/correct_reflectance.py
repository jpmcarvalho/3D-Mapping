#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fábio Azevedo & João Pedro Carvalho
"""

import argparse
import exiftool

import os, glob
import micasense.capture as capture

import cv2
import numpy as np
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils

import micasense.metadata as metadata
from datetime import datetime, timedelta
from designThermalStable import deignThermal

from os import path

import imageio

# Argument parser
parser = argparse.ArgumentParser(description='Correct Micasense reflectance images. Update EXIF information.')
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='data_raw', help='File directory')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
parser.add_argument("-o", "--outdir", type=str, default='images', help='Output file directory.')
parser.add_argument("-s", "--splitImages", type=int, default=150, help='Maximum number of images per group.')
parser.add_argument("--isMultispectral", help="Preprocess Multispectral images", action="store_true")
parser.add_argument(      "--thermal", action="store_true", help='Thermal.')
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


def create_image_groups(images, n_related_images, split_size):
    file_groups = open(os.path.join(args.base, args.uid, "image_groups.txt"), "w")
    init_char = ord('A')
    group_size = (len(images) / (int(len(images) / split_size + 1))) / n_related_images

    for i in range(int(len(images) / n_related_images)):
        for j in range(n_related_images):
            idx = int(i / group_size)
            file_groups.write(os.path.basename(images[i * n_related_images + j]) + ' ' + chr(init_char + idx) + '\n')

    file_groups.close()


imagePath = os.path.join(args.base, args.uid, args.dir)

if not args.isMultispectral:
    os.system("cp " + os.path.join(imagePath, 'Mission') + "/* " + outdir)
    # create_image_groups(sorted(glob.glob(os.path.join(outdir,'IMG_*.tif'))), 1, args.splitImages)
    exit(0)

imageNames = glob.glob(os.path.join(imagePath, 'Mission', '*.tif'))
md = metadata.Metadata(imageNames[0])
cam_model = md.get_item('EXIF:Model')
print("Camera model:", cam_model)
# If images are not multispectral, only copy them to the appropriate folder and exit
if cam_model == 'RedEdge-P':
    os.system("cp " + os.path.join(imagePath, 'Mission') + "/* " + outdir)
    # create_image_groups(sorted(glob.glob(os.path.join(outdir,'IMG_*.tif'))), 1, args.splitImages)
    exit(0)

panelNames = None

panelNames = glob.glob(os.path.join(imagePath, 'Panel', 'IMG_0000_*.tif'))
panelNames = sorted(panelNames)

isAltum = False
if len(panelNames) > 5:
    isAltum = True
    panelNames = panelNames[:5]

print(panelNames)
print(outdir)
print(imagePath)
print(args.base)

# Allow this code to align both radiance and reflectance images; bu excluding
# a definition for panelNames above, radiance images will be used
# For panel images, efforts will be made to automatically extract the panel information
# but if the panel/firmware is before Altum 1.3.5, RedEdge 5.1.7 the panel reflectance
# will need to be set in the panel_reflectance_by_band variable.
# Note: radiance images will not be used to properly create NDVI/NDRE images below.
# NOTE2: The panel images MUST have the name IMG_0000_*.tif
if panelNames is not None:
    panelCap = capture.Capture.from_filelist(panelNames)
else:
    panelCap = None

# Get Mission files
for root, dirs, files in os.walk(os.path.join(imagePath, 'Mission')):
    if len(sorted(filter(lambda f: f.endswith("_6.tif"), files))) > 0:
        isAltum = True
    if isAltum is not True:
        number_files = int(len(files) / 5)
    else:
        number_files = int(len(files) / 6)
    first_file = sorted(filter(lambda f: f.endswith("_1.tif"), files))[0]

# Get number of first image
img_offset = int(first_file[4:8])
print("First file:", first_file)

# Extract panel calibrations
if panelCap is not None:
    panelCap.detect_panels()

    if panelCap.panel_albedo() is not None:
        panel_reflectance_by_band = panelCap.panel_albedo()
    else:
        panel_reflectance_by_band = [0.519, 0.521, 0.52, 0.519, 0.518]  # RedEdge band_index order
    
    print(panel_reflectance_by_band)
    panel_irradiance = panelCap.panel_irradiance(
        panel_reflectance_by_band)  # TODO need to add a try catch in this function
    img_type = "reflectance"
else:
    if imageCap.dls_present():
        img_type = 'reflectance'
        imageCap.compute_undistorted_reflectance(panel_irradiance)
    else:
        img_type = "radiance"

# Create list of images
im_num = list(range(1, number_files + 1))

# While there is images to treat
while (len(im_num)):
    print(len(im_num))
    imageNames = None
    imageCap = None

    # Get next image
    img_number = im_num[0]
    img_str = str(img_number + img_offset - 1).zfill(4)

    # Global path
    imageNames = glob.glob(os.path.join(imagePath, 'Mission', 'IMG_' + img_str + '_*.tif'))

    if len(imageNames) > 5: #isAltum:
        imageNames = sorted(imageNames)
        imageNames = imageNames[:len(imageNames)-1]

    # Verify if there is 5 bands of the capture
    if len(imageNames) < 5:
        print("Not enough files!")
        number_files += 1
        del im_num[0]
        im_num.append(number_files)
        continue

    # Compute reflectance
    imageCap = capture.Capture.from_filelist(imageNames)

    # For each band
    for i in range(1, len(imageNames) + 1):
        img = imageCap.images[i - 1]

        # Get original image metadata
        md = metadata.Metadata(imageNames[i - 1])

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

        # Calculate corrected image
        # corr_img = img.undistorted_reflectance().astype('float32')
        corr_img = img.reflectance(irradiance=panel_irradiance[i - 1]).astype('float32')

        # Save corrected image uint16 TIFF
        imtype = 'tif'
        band_filename = os.path.join(imagePath, 'Mission', 'IMG_' + img_str + '_' + str(i) + '.tif')
        filename = os.path.join(outdir, 'IMG_' + img_str + '_' + str(i) + '.' + imtype)
        imageio.imwrite(filename, (255 * 256 * corr_img).astype('uint16'))

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

        # if os.path.exists(filename+"_original"):
        #   os.remove(filename+"_original")
        if i == 5 and isAltum and args.thermal:
            band_filename = os.path.join(imagePath, 'Mission', 'IMG_' + img_str + '_' + str(i+1) + '.tif')
            filename = os.path.join(outdir, 'IMG_' + img_str + '_' + str(i+1) + '.' + imtype)
            # os.system('cp ' + band_filename + ' ' + filename)
            deignThermal(band_filename, filename)

    # Remove images from list
    del im_num[0]

if os.path.exists(os.path.join(outdir)):
        fileList = glob.glob(os.path.join(outdir, "*_original"))
        # Iterate over the list of filepaths & remove each file.
        for filePath in fileList:
            try:
                os.remove(filePath)
            except OSError:
                print("Error while deleting file")
# create_image_groups(sorted(glob.glob(os.path.join(outdir,'IMG_*.tif'))), 5, args.splitImages)
