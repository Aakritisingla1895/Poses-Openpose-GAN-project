"""
convert bvh file into a dataset for further processing by the mocap_dataset class
this is a pre-requisite for training any machine learning systems
currently, the code is not able to automatically determine the correct order of euler rotations
used in the bvh file
for this reason, this order needs to be specified by the user
for bvh data exported from Captury Studio the order is x, y, z
for bvh data exported from MotionBuilder the order is z, x, y
"""

import argparse
from bvh_parsers import BVH_Parser
from bvh_tools import *
from dataset_tools import DatasetTools
import pickle


parser = argparse.ArgumentParser(description='convert bvh file into mocap file')

parser.add_argument('--input', type=str, nargs='+',
                    help='input bvh file')
parser.add_argument('--output', type=str, nargs='+',
                    help='output mocap file')

args = parser.parse_args()

input_file_name = args.input[0]
output_file_name = args.output[0]


bvh_tools = BVH_Tools()

#captury euler rotation sequence
#bvh_tools.euler_sequence = [0, 1, 2] # x, y, z

#motion builder euler rotation sequence
bvh_tools.euler_sequence = [2, 0, 1] # z, x, y

#Rokoko Suit euler rotation sequence
#bvh_tools.euler_sequence = [1, 0, 2] # y, x, z

skeletons, skeleton_frames = bvh_tools.parse_bvh_file(input_file_name)
datasets = bvh_tools.create_datasets()

# store as pickle file
if output_file_name.endswith(".p"):
    pickle.dump( datasets, open( output_file_name, "wb" ) )

# store as json file
if output_file_name.endswith(".json"):
    datatools = DatasetTools()
    datatools.dataset = datasets
    datatools.save_dataset(output_file_name)
