#!/usr/bin/python3

# Basic check
import sys
if sys.version_info.major < 3:
    print("Ups! ODM needs to run with Python 3. It seems you launched it with Python 2. Try using: python3 run.py ... ")
    sys.exit(1)

import os
from opendm import log
from opendm import config
from opendm import system
from opendm import io
from opendm.progress import progressbc
from opendm.utils import get_processing_results_paths, rm_r
from opendm.arghelpers import args_to_dict, save_opts, compare_args, find_rerun_stage

from stages.odm_app import ODMApp

def odm_version():
    try:
        with open("VERSION") as f:
            return f.read().split("\n")[0].strip()
    except:
        return "?"

if __name__ == '__main__':
    args = config.config()

    log.ODM_INFO('Initializing ODM %s - %s' % (odm_version(), system.now()))

    progressbc.set_project_name(args.name)
    args.project_path = os.path.join(args.project_path, args.name)
    print(args.project_path)

    log.ODM_INFO("Writing args.")
    file = open(os.path.join(args.project_path, "args.txt"), 'w+')
    file.write(str(args).replace(", ", ",\n"))
    file.close()

    progressbc.set_project_path(args.project_path)
    progressbc.send_update(0)

    if not io.dir_exists(args.project_path):
        log.ODM_WARNING('Directory %s does not exist. Creating it now.' % args.name)
        system.mkdir_p(os.path.abspath(args.project_path))

    opts_json = os.path.join(args.project_path, "options.json")
    auto_rerun_stage, opts_diff = find_rerun_stage(opts_json, args, config.rerun_stages, config.processopts)
    if auto_rerun_stage is not None and len(auto_rerun_stage) > 0:
        log.ODM_INFO("Rerunning from: %s" % auto_rerun_stage[0])
        args.rerun_from = auto_rerun_stage

    # Print args
    args_dict = args_to_dict(args)
    log.ODM_INFO('==============')
    for k in args_dict.keys():
        log.ODM_INFO('%s: %s%s' % (k, args_dict[k], ' [changed]' if k in opts_diff else ''))
    log.ODM_INFO('==============')
    

    # If user asks to rerun everything, delete all of the existing progress directories.
    if args.rerun_all:
        log.ODM_INFO("Rerun all -- Removing old data")
        for d in [os.path.join(args.project_path, p) for p in get_processing_results_paths()] + [
                  os.path.join(args.project_path, "odm_meshing"),
                  os.path.join(args.project_path, "opensfm"),
                  os.path.join(args.project_path, "odm_texturing_25d"),
                  os.path.join(args.project_path, "odm_filterpoints"),
                  os.path.join(args.project_path, "submodels"),
                  os.path.join(args.project_path, "reflectance"),
                  os.path.join(args.project_path, "images_backup"),
                  os.path.join(args.project_path, "images"),
                  os.path.join(args.project_path, "maps"),
                  os.path.join(args.project_path, "odm_orthophoto_1"),
                  os.path.join(args.project_path, "odm_orthophoto_2"),
                  os.path.join(args.project_path, "odm_orthophoto_3"),
                  os.path.join(args.project_path, "odm_orthophoto_4"),
                  os.path.join(args.project_path, "odm_orthophoto_5"),]:
            rm_r(d)

    # Set the control variable for the RGB vs Multispectral input images
    mult_str = ""
    if args.isMultispectral:
        mult_str += " --isMultispectral"

    isThermal_str = ""
    if args.thermal:
        isThermal_str += " --thermal "

    if args.hasLidarPLY:
        inputDir = os.path.join(args.project_path, 'data_raw', 'Lidar', 'point_cloud.ply')

        if not os.path.isfile(inputDir):
            log.ODM_WARNING(
                "Not found a valid point cloud file in: %s." % inputDir
            )
            exit(0)

    # Call the pre-processing procedure
    if args.mainModel:
        os.system('touch ' + str(args.project_path) + '/startMap.txt')
        os.system("python3 correct_reflectance.py --uid " + args.uid + mult_str + isThermal_str)
        os.system('touch ' + str(args.project_path) + '/startReflectance.txt')
    # if args.isMultispectral:
    #    os.system('mkdir -p ' + str(args.project_path)+'/reflectance')
    #   os.system('cp ' + str(args.project_path)+'/images/* ' + str(args.project_path)+'/reflectance/')
    #  os.system("rm -rf " + " ".join([quote(os.path.join(args.project_path, "images"))]))
    #  os.system('mkdir -p ' + str(args.project_path) + '/images')
    #  os.system('cp ' + str(args.project_path) + '/reflectance/*_1* ' + str(args.project_path) + '/images/')
    if args.flir:
        os.system("python3 extractor.py --uid " + args.uid)
    allBands = 1

    # while allBands < 6:
    app = ODMApp(args)
    retcode = app.execute()

    if retcode == 0:
        save_opts(opts_json, args)
    
    if args.mainModel:
        # Parse the input arguments
        generate_str = ""
        if args.ndvi:
            generate_str += "--ndvi "
        if args.ndre:
            generate_str += "--ndre "
        if args.rgb:
            generate_str += "--rgb "
        if args.cir:
            generate_str += "--cir "
        if args.ndwi:
            generate_str += "--ndwi "
        if args.thermal:
            generate_str += "--thermal "
        if args.flir:
            generate_str += "--flir "

        # Call the generation of the final tif and png files
        # os.system("python3 homographyBands.py " + generate_str + "--uid " + args.uid + " " + mult_str)
        os.system("python3 generate_indexes.py " + generate_str + "--uid " + args.uid + " " + mult_str)

        if args.isMultispectral:
            os.system("python3 paint_PCD.py -id " + args.uid + " " + generate_str)
            if args.ndvi:
                os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_modelNDVIcolorFixed.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_NDVI'))
            if args.ndre:
                os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_modelNDREcolorFixed.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_NDRE'))
            if args.rgb:
                os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_modelRGBcolorFixed.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_RGB'))
            if args.cir:
                os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_modelCIRcolorFixed.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_CIR'))
            if args.ndwi:
                os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_modelNDWIcolorFixed.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_NDWI'))
            os.system("rm -rf " + os.path.join(args.project_path, 'entwine_pointcloud-tmp'))
        elif args.flir:
            os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_model.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_Flir'))

            os.system("rm -rf " + os.path.join(args.project_path, 'entwine_pointcloud-tmp'))
        else:
            os.system('/code/SuperBuild/install/bin/entwine build --threads 8 --tmp ' + os.path.join(args.project_path, 'entwine_pointcloud-tmp') + ' -i ' + os.path.join(args.project_path, 'odm_georeferencing/odm_georeferenced_model.laz') + ' -o ' + os.path.join(args.project_path, 'entwine_pointcloud', 'entwine_pointcloud_RGB'))
            os.system("rm -rf " + os.path.join(args.project_path, 'entwine_pointcloud-tmp'))





    # Do not show ASCII art for local submodels runs
    if retcode == 0 and not "submodels" in args.project_path:
        log.ODM_INFO('HHHHHHHHH     HHHHHHHHH                     iiii     ffffffffffffffff                    ')
        log.ODM_INFO('H:::::::H     H:::::::H                    i::::i   f::::::::::::::::f                   ')
        log.ODM_INFO('H:::::::H     H:::::::H                     iiii   f::::::::::::::::::f                  ')
        log.ODM_INFO('HH::::::H     H::::::HH                            f::::::fffffff:::::f                  ')
        log.ODM_INFO('  H:::::H     H:::::H      eeeeeeeeeeee   iiiiiii  f:::::f       ffffffuuuuuu    uuuuuu  ')
        log.ODM_INFO('  H:::::H     H:::::H    ee::::::::::::ee i:::::i  f:::::f             u::::u    u::::u  ')
        log.ODM_INFO('  H::::::HHHHH::::::H   e::::::eeeee:::::eei::::i f:::::::ffffff       u::::u    u::::u  ')
        log.ODM_INFO('  H:::::::::::::::::H  e::::::e     e:::::ei::::i f::::::::::::f       u::::u    u::::u  ')
        log.ODM_INFO('  H:::::::::::::::::H  e:::::::eeeee::::::ei::::i f::::::::::::f       u::::u    u::::u  ')
        log.ODM_INFO('  H::::::HHHHH::::::H  e:::::::::::::::::e i::::i f:::::::ffffff       u::::u    u::::u  ')
        log.ODM_INFO('  H:::::H     H:::::H  e::::::eeeeeeeeeee  i::::i  f:::::f             u::::u    u::::u  ')
        log.ODM_INFO('  H:::::H     H:::::H  e:::::::e           i::::i  f:::::f             u:::::uuuu:::::u  ')
        log.ODM_INFO('HH::::::H     H::::::HHe::::::::e         i::::::if:::::::f            u:::::::::::::::uu')
        log.ODM_INFO('H:::::::H     H:::::::H e::::::::eeeeeeee i::::::if:::::::f             u:::::::::::::::u')
        log.ODM_INFO('H:::::::H     H:::::::H  ee:::::::::::::e i::::::if:::::::f              uu::::::::uu:::u')
        log.ODM_INFO('HHHHHHHHH     HHHHHHHHH    eeeeeeeeeeeeee iiiiiiiifffffffff                uuuuuuuu  uuuu')

        log.ODM_INFO('ODM app finished - %s' % system.now())

        os.system('mkdir ' + str(args.project_path) + '/images')
        os.system('touch ' + str(args.project_path) + '/finishODM.txt')
    else:
        exit(retcode)