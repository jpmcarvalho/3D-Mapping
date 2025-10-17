#!/bin/bash

PATH1=$1
PATH2=$2

exiftool -tagsfromfile "$PATH1" -all:all -xmp -exif -b "$PATH2" -ext jpg >/dev/null;
#echo "y" | exiftool -delete_original $PATH2 >/dev/null;
sleep 0.5
#rm -r "$PATH2"_original >/dev/null;