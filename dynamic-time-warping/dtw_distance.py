import os
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--first_id', type=int, default=1)
# parser.add_argument('--second_id', type=int, default=2)
sorting_parser = parser.add_mutually_exclusive_group(required=False)
sorting_parser.add_argument('--sorting', dest='sorting', action='store_true')
sorting_parser.add_argument('--no-sorting', dest='sorting', action='store_false')
parser.set_defaults(sorting=True)

args = parser.parse_args()

# first_id = args.first_id
# second_id = args.second_id
sorting = args.sorting

def compare_two_matrices(first_path, second_path):
    matrix_1 = np.array(np.loadtxt(fname=first_path))
    matrix_2 = np.array(np.loadtxt(fname=second_path))
    
    distance, path = fastdtw(matrix_1, matrix_2, dist=euclidean)
    
    max_length = matrix_1.shape[0] if matrix_1.shape[0] > matrix_2.shape[0] else matrix_2.shape[0]
    norm_distance = (max_length - distance) / max_length

    return norm_distance

directory = 'matrices'
filenames = []
for filename in os.listdir(directory):
    filenames.append(filename)

filenames.sort()

distances = []
for i, first_file in enumerate(filenames):
    for j in range(i, len(filenames)):
        if i != j:
            first_path = directory + '/' + first_file
            second_path = directory + '/' + filenames[j]
            norm_distance = compare_two_matrices(first_path, second_path)
            distances.append({'first': first_path, 'second': second_path, 'distance': norm_distance})
            print("Calulating distance between %s and %s" %(first_path, second_path))
    print('')


if sorting:
    distances.sort(key=lambda x: x['distance'])

for distance_object in distances:
    print("Distance between %s and %s is: %.5f" %(distance_object['first'], distance_object['second'], distance_object['distance']))