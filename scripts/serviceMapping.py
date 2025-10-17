#!flask/bin/python
from flask import Flask
from flask_cors import CORS
import psutil

import os
import subprocess

from flask import request
import json

psutil.process_iter(attrs=None, ad_value=None)

app = Flask(__name__)
CORS(app)

@app.route('/postjson', methods=['GET','POST'])
def post():

    # Requester ID
    uID = request.json['id']

    # Parse the intended bands to generate
    band_str = request.json['bands']

    # Fill the RGB vs Multispectral variable
    mult_str = ""
    rad_str = " none "
    if request.json['isMultispectral']:
        mult_str += " --isMultispectral "
        #rad_str = " camera "

    # Request for 3D model
    is3d_str = " "
    #if request.json['is3d'] and not request.json['isMultispectral']:
     #   is3d_str += " --pc-ept "



    # Call the running script
    #os.system("bash /code/run.sh --uid " + str(uID) + " --name code/data/" + str(uID) +
     #         " --mesh-size 100000 --radiometric-calibration " + rad_str +
      #        " --max-concurrency 4 --split 2000 --split-overlap 150 --merge all --mainModel --orthophoto-cutline --crop 0.5 " + #--resize-to -1 " +
       #       band_str + mult_str + is3d_str + " --min-num-features 16000 --rerun-all &")

    os.system("bash /code/run.sh --uid " + str(uID) + " --name code/data/" + str(uID) +
              " --mesh-size 100000 --radiometric-calibration " + rad_str +
              "--max-concurrency 8 --feature-type sift --min-num-features 24000 --split 2000 --split-overlap 150 --merge all --mainModel --orthophoto-cutline --crop 0.5 " +  # --resize-to -1 " +
              band_str + mult_str + is3d_str + " --rerun-all &")

    return 'Start Running Mapping'

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

app.run(host='0.0.0.0', port=5001)


