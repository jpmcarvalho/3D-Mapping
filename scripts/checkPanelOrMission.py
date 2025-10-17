#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import micasense.capture as capture
import micasense.metadata as metadata
import cv2
import zipfile
from pipes import quote
import requests

# Argument parser
parser = argparse.ArgumentParser(description='Correct Micasense reflectance images. Update EXIF information.')
#  parser.add_argument("-b", "--base", type=str, default='/Users/jpc/Documents/PDMFC/data', help='Base directory')
parser.add_argument("-b", "--base", type=str, default='/code/data', help='Base directory')
parser.add_argument("-d", "--dir", type=str, default='data_raw', help='File directory')
parser.add_argument("-u", "--uid", type=str, default='1', help='User ID')
parser.add_argument("--isMultispectral", help="Preprocess Multispectral images", action="store_true")
parser.add_argument("-v", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("Version 0.1.")
    exit()

zipPath = os.path.join(args.base, args.uid, 'zip')
missionPath = os.path.join(args.base, args.uid, args.dir, 'Mission')
panelPath = os.path.join(args.base, args.uid, args.dir, 'Panel')


def saveAnyFileToPNG(filesName, isMission=True):
    for pathFile in filesName:
        img = cv2.imread(pathFile, cv2.IMREAD_UNCHANGED)
        headAndTail = os.path.split(pathFile)
        if isMission:
            fMission.write(str(headAndTail[1].split('.')[0]) + ';')
        else:
            fPanel.write(str(headAndTail[1].split('.')[0]) + ';')
        saveNewPath = headAndTail[0] + 'R/' + str(headAndTail[1].split('.')[0])
        print(saveNewPath)
        cv2.imwrite(saveNewPath + '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(saveNewPath + '.webp', img, [cv2.IMWRITE_WEBP_QUALITY, 75])


def saveRGBImagesInRightPlace():
    for root, dirs, files in os.walk(os.path.join(zipPath)):
        files = sorted(files)
        while len(files):
            file =files[0]
            imageName = os.path.join(root, file)
            extension = imageName.split('.')[1].lower()
            if imageName is not None:
                if extension == 'jpg' or extension == 'png' or extension == 'tif' or extension == 'jpeg' or \
                        extension == 'tiff':
                    # print("mv '" + root + '/' + file + "' " + missionPath + '/')
                    os.system("mv '" + root + '/' + file + "' " + missionPath + '/')
                    saveAnyFileToPNG(sorted(glob.glob(os.path.join(missionPath, file))))
            del files[0]


def saveMultispectralImagesInRightPlace():
    for root, dirs, files in os.walk(os.path.join(zipPath)):
        files = sorted(files)
        while len(files):
            file = files[0]
            img_str = (file[4:8]).zfill(4)

            imageNames = glob.glob(os.path.join(root, 'IMG_' + img_str + '_*.tif'))
            imageNames = sorted(imageNames)

            if len(imageNames) < 5:
                print("Not enough files-> " + str(os.path.join(root, file)))
                if len(imageNames) > 0:
                    del files[0:len(imageNames)]
                else:
                    del files[0:len(file)]
                continue

            pathToRead = None
            isMission = True
            if imageNames is not None:
                md = metadata.Metadata(imageNames[0])
                cam_model = md.get_item('EXIF:Model')
                if cam_model == 'RedEdge-P':
                    print('mv ' + root + '/IMG_' + img_str + '_*.tif ' + missionPath + '/')
                    os.system("mv '" + root + '/IMG_' + img_str + "_1.tif' " + missionPath + '/')
                    os.system("mv '" + root + '/IMG_' + img_str + "_2.tif' " + missionPath + '/')
                    os.system("mv '" + root + '/IMG_' + img_str + "_3.tif' " + missionPath + '/')
                    os.system("mv '" + root + '/IMG_' + img_str + "_4.tif' " + missionPath + '/')
                    os.system("mv '" + root + '/IMG_' + img_str + "_5.tif' " + missionPath + '/')
                    pathToRead = missionPath
                else:
                    panelCap = capture.Capture.from_filelist(imageNames)
                    if panelCap.panels_in_all_expected_images():
                        print('mv ' + root + '/IMG_' + img_str + '_*.tif ' + panelPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_1.tif' " + panelPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_2.tif' " + panelPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_3.tif' " + panelPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_4.tif' " + panelPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_5.tif' " + panelPath + '/')
                        pathToRead = panelPath
                        isMission = False
                    else:
                        print('mv ' + root + '/IMG_' + img_str + '_*.tif ' + missionPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_1.tif' " + missionPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_2.tif' " + missionPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_3.tif' " + missionPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_4.tif' " + missionPath + '/')
                        os.system("mv '" + root + '/IMG_' + img_str + "_5.tif' " + missionPath + '/')
                        pathToRead = missionPath

            saveAnyFileToPNG(sorted(glob.glob(os.path.join(pathToRead, 'IMG_' + img_str + '_*.tif'))), isMission)
            del files[0:len(imageNames)]


allMissionImagesPath = os.path.join(args.base, args.uid, 'allMissionImages.txt')
allPanelImagesPath = os.path.join(args.base, args.uid, 'allPanelImages.txt')
zipFilePath = os.path.join(zipPath, 'zip.zip')

if os.path.exists(allMissionImagesPath):
    os.system("rm -rf " + " ".join([quote(allMissionImagesPath), ]))

if os.path.exists(allPanelImagesPath):
    os.system("rm -rf " + " ".join([quote(allPanelImagesPath), ]))

if os.path.exists(zipFilePath):
    with zipfile.ZipFile(zipFilePath, 'r') as zip_ref:
        zip_ref.extractall(zipPath)

    fMission = open(allMissionImagesPath, 'w')
    fPanel = open(allPanelImagesPath, 'w')

    if args.isMultispectral:
        saveMultispectralImagesInRightPlace()
    else:
        saveRGBImagesInRightPlace()
    fMission.close()
    fPanel.close()

    os.system("rm -rf " + " ".join([quote(zipPath), ]))
    os.system('mkdir ' + str(zipPath))

    url = 'http://backend:3000/api/v1/map/finished/separation/' + args.uid
    receive = requests.post(url)
else:
    print(zipFilePath + ' does not exist!')
