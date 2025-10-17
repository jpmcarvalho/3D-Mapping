#!flask/bin/python
from flask import Flask, send_file
from flask_cors import CORS
import psutil

import os
import subprocess

from flask import request
import json
import time

from volume.volume import calc_volume

psutil.process_iter(attrs=None, ad_value=None)

app = Flask(__name__)
CORS(app)
last_call_time = {}
ENTRY_TTL = 600  # delete entries with more than 10min

@app.route('/postjson', methods=['GET','POST'])
def post():

    # Requester ID
    uID = request.json['id']

    now = time.time()
    expired = [key for key, val in last_call_time.items() if now - val > ENTRY_TTL]
    for key in expired:
        del last_call_time[key]


    now = time.time()
    last_value = last_call_time.get(uID, 0)
    if now - last_value < 10:  # 10 seconds threshold
        return 'Request too frequent', 429  # HTTP 429 Too Many Requests
    last_call_time[uID] = now

    # Parse the intended bands to generate
    band_str = request.json['bands']

    # Fill the RGB vs Multispectral variable
    mult_str = ""
    rad_str = " none "
    if request.json['isMultispectral']:
        mult_str += " --isMultispectral "
        #rad_str = " camera "

    flir_str = ""
    try:
        if request.json['flir']:
            flir_str += " --flir "
    except:
        pass

    # Request for 3D model
    is3d_str = " "
    #if request.json['is3d'] and not request.json['isMultispectral']:
     #   is3d_str += " --pc-ept "

    lidar_str = ""
    if 'hasLidarPLY' in str(request.json):
        if request.json['hasLidarPLY'] == True:
            lidar_str += " --hasLidarPLY "

    # Call the running script
    #os.system("bash /code/run.sh --uid " + str(uID) + " --name code/data/" + str(uID) +
     #         " --mesh-size 100000 --radiometric-calibration " + rad_str +
      #        " --max-concurrency 4 --split 2000 --split-overlap 150 --merge all --mainModel --orthophoto-cutline --crop 0.5 " + #--resize-to -1 " +
       #       band_str + mult_str + is3d_str + " --min-num-features 16000 --rerun-all &")
#--flir --use-3dmesh
    os.system("bash /code/run.sh --uid " + str(uID) + " --name code/data/" + str(uID) +
              " --auto-boundary  --dem-euclidean-map --pc-rectify --use-3dmesh " +
              " --dem-resolution 1 --mesh-size 300000 --radiometric-calibration " + rad_str +
              " --max-concurrency 16 --feature-type sift --min-num-features 24000 --mainModel --orthophoto-cutline --dtm --dsm --crop 0.5 --cog " +
              " --orthophoto-resolution 0.1 --feature-quality ultra --pc-quality high " +  # --resize-to -1 "
              " --mesh-octree-depth 12 " +
              f" {band_str} {mult_str} {is3d_str} {flir_str} {lidar_str} 2>&1 | tee -a /code/data/{str(uID)}/output.log &")


    return 'Start Running Mapping\n'

@app.route('/postjsonZip', methods=['GET','POST'])
def postjsonZip():
    # Requester ID
    uID = request.json['id']

    # Fill the RGB vs Multispectral variable
    mult_str = ""
    if request.json['isMultispectral']:
        mult_str += " --isMultispectral "

    os.system("python3 /code/checkPanelOrMission.py --uid " + str(uID) + mult_str + " &")
    return 'Starting extract zipping'

@app.route('/stopProcess', methods=['GET','POST'])
def stopAProcess():
    uID = request.json['id']
    if os.path.exists('/code/process_files/' + str(uID)):
        f = open('/code/process_files/' + str(uID), 'r')
        os.system('bash /code/kill_process.sh ' + str(f.read()))
        os.remove('/code/process_files/' + str(uID))
    return 'Stop Mapping'

@app.route('/deleteProcess', methods=['GET','POST'])
def deleteAProcess():
    uID = request.json['id']
    if os.path.exists('/code/process_files/' + str(uID)):
        os.remove('/code/process_files/' + str(uID))
    return 'Delete Mapping Process'


@app.route('/orthophoto/tile', methods=['GET','POST'])
def ortophotoTile():
    # Requester ID
    uID = request.json['id']
    z = request.json['z']
    x = request.json['x']
    y = request.json['y']
    rescale = request.json['rescale']
    index = request.json['index']

    # Fill the remaining variables
    mult_str = ""
    if z:
        mult_str += " -z " + z

    if x:
        mult_str += " -x " + x

    if y:
        mult_str += " -y " + y

    if rescale:
        mult_str += " --rescale " + rescale

    if index:
        mult_str += " --index " + index

    # os.system("python3 /code/orthophotoTile.py --uid " + str(uID) + mult_str + " &")
    p = subprocess.Popen("python3 scripts/orthophotoTile.py --uid " + str(uID) + mult_str + " &", shell=True)
    result = p.wait()

    basedir = "/code/data"
    odm_dir = "maps"

    file = os.path.join(basedir, uID, odm_dir, "output_{0}_{1}_{2}_{3}.png".format(z, x, y, index))

    maxTries = 200
    tryCounter = 0

    while not os.path.exists(file) and tryCounter < maxTries:
        time.sleep(0.1)
        tryCounter += 1

    
    return str(result)


@app.route('/calculate/volume/', methods=['GET', 'POST'])
def calcVolume():
    print(request.json)
    points = request.json['coordinates']
    method = request.json['method']
    uID = request.json['id']
    
    # print(area)
    # print(method)
    # print(uID)
    
    # area = serializer['area'].value
    # method = serializer['method'].value
    # points = [coord for coord in area['geometry']['coordinates'][0]]
    print(points)

    basedir = "/code/data"
    # basedir = "/home/draco/Downloads"
    odm_dir = "maps"

    dsm = os.path.join(basedir, uID, odm_dir, "dsm.tif")
    # dsm = os.path.join(basedir, "dsm.tif")

    volume = calc_volume(dsm, points, 4326, base_method=method)

    print(volume)

    return str(volume)




app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
