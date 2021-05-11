from os import listdir, mkdir
from shutil import rmtree
from os.path import join

import transform2BEV

base_dir = './'
cal_dir = '../data_road_2/testing/calib/'
vis_dir = 'pred_mask/'
submit_dir = 'to_submit/'

try:
    rmtree(submit_dir)
except OSError:
    pass
mkdir(submit_dir)


testData_pathToCalib = cal_dir
outputDir_perspective = vis_dir
outputDir_bev = submit_dir

inputFiles = join(outputDir_perspective, '*.png')
transform2BEV.main(inputFiles, testData_pathToCalib, outputDir_bev)

# check
assert len([f for f in listdir(vis_dir) if f.endswith('.png')]) == 290
assert len(listdir(cal_dir)) == 290
assert len(listdir(submit_dir)) == 290
