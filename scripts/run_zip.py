#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: João Pedro Matos Carvalho
"""

import glob
import argparse
import os
import shutil

import sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages/')

import cv2 

# Argument parser
parser = argparse.ArgumentParser(description='Generate NDVI from odm_orthophoto.')
parser.add_argument("-b", "--base",    type=str,   default='/code/data',         help='Base directory')
parser.add_argument("-d", "--dir",     type=str,   default='maps',     help='Folder directory')
parser.add_argument("-u", "--uid",     type=str,   default='1',                  help='User ID')
parser.add_argument("-v", "--version",         help="show program version",              action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

# Generate input file and output directory path strings
folder = os.path.join(args.base, args.uid, args.dir)
str_find = os.path.join(args.base, args.uid, args.dir,'map')

if not os.path.exists(folder):
    print("Folder " + folder + " not found!")
    exit()

width = 400
for filename in glob.glob(folder+'/*.png'):
	# Generate png images
	out_name = filename[:len(str_find)] + 'r' + filename[len(str_find):]
	img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
	ratio = float(width) / float(img.shape[1])
	height = int(img.shape[0] * ratio)
	resized = cv2.resize(img, (width,height), interpolation=cv2.INTER_AREA)
	cv2.imwrite(os.path.join(folder, out_name), resized)

# Move dstm, dtm and 3d info to maps folder
odm_texturing = os.path.join(args.base, args.uid, args.dir, "odm_texturing")
os.makedirs(odm_texturing, 0o777, True)
os.system("mv " + args.base + "/" + args.uid + "/odm_texturing/* " + odm_texturing)
os.system("mv " + args.base + "/" + args.uid + "/odm_dem/dsm.tif " + folder)
os.system("mv " + args.base + "/" + args.uid + "/odm_dem/dtm.tif " + folder)
os.system("mv " + args.base + "/" + args.uid + "/odm_orthophoto/odm_orthophoto.tif " + folder)

print("Generating ZIP file!")
shutil.make_archive(base_name=folder, format='zip', root_dir=folder)
os.system("mv " + folder + ".zip " + folder)
