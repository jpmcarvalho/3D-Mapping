#!/usr/bin/python

import cv2 as cv2
import numpy as np
import imageio
import os
from osgeo import gdal

filePath = '/home/draco/Documents/colorThermal/map/'

imgThermal = cv2.imread(os.path.join(filePath, 'mapRGB.png'), -1)

imgThermalColor = cv2.applyColorMap(imgThermal, cv2.COLORMAP_JET)
alpha_channel = np.zeros(imgThermal.shape, dtype=imgThermalColor.dtype)
alpha_channel[imgThermal != 0] = 255
img_BGRA = cv2.merge((imgThermalColor, alpha_channel))
cv2.imwrite(os.path.join(filePath, 'mapRGBA.png'), img_BGRA)

imgResult = cv2.imread(os.path.join(filePath, 'mapRGB.tif'), -1)
maxValueTif = (2**16)-1
imgResult = (imgResult).astype('uint8')
alpha_channel_result = np.zeros(imgResult.shape, dtype=imgResult.dtype)
alpha_channel_result[imgResult != 0] = 255
imgResultColor = cv2.applyColorMap(imgResult, cv2.COLORMAP_JET)

ds = gdal.Open(os.path.join(filePath, 'mapRGB.tif'), gdal.GA_ReadOnly)
driver = gdal.GetDriverByName("GTiff")
options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
thermal =  ds.GetRasterBand(1).ReadAsArray()

[cols, rows] = thermal.shape
outdata = driver.Create(os.path.join(filePath, 'mapRGBA.tif'), rows, cols, 4, gdal.GDT_Byte, options=options)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input

outdata.GetRasterBand(1).WriteArray(imgResultColor[:,:,2])
outdata.GetRasterBand(2).WriteArray(imgResultColor[:,:,1])
outdata.GetRasterBand(3).WriteArray(imgResultColor[:,:,0])
outdata.GetRasterBand(4).WriteArray(alpha_channel_result)
outdata.GetRasterBand(1).SetNoDataValue(255)##if you want these values transparent
outdata.GetRasterBand(2).SetNoDataValue(255)##if you want these values transparent
outdata.GetRasterBand(3).SetNoDataValue(255)##if you want these values transparent
outdata.GetRasterBand(4).SetNoDataValue(0)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
