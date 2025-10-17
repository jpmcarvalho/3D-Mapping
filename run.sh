#!/bin/bash

RUNPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RUNPATH/SuperBuild/install/lib
python3 $RUNPATH/run.py "$@" &

# Get value of --name from second set of arguments
delimiter=--uid
IFS=' ' read -ra filename <<< "${@#*"$delimiter"}"
nameFile=${filename[0]}
echo $filename

if [ "$filename" = "--help" ]; then
    exit 0
fi

processFiles="/code/process_files/${nameFile}"
echo $! >> "${processFiles}"