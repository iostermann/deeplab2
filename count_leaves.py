import collections
import os
import tempfile

from absl import app
from absl import flags
from joblib import parallel_backend
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import glob
from os import path
from PIL import Image

flags.DEFINE_string(
  'input_images',
  default=None,
  help='images to run inference on. Stored as .npy numpy arrays'
)

flags.DEFINE_string(
    'output_csv_name',
    default='test.csv',
    help='name of csv to output data to'
)

FLAGS = flags.FLAGS

stem = np.dot([0, 0, 0], [1, 256, 256 * 256])
background = np.dot([255, 255, 255], [1, 256, 256 * 256])

def process_side_rotations(x):
    regex = "side_Skeleton_" + str(x) + "*_image.npy"
    average = count_average_leaves_for_plant(regex)
    return average

def process_top_rotations(x):
    regex = "top_Skeleton_" + str(x) + "*_image.npy"
    average = count_average_leaves_for_plant(regex)
    return average

def count_average_leaves_for_plant(regex):
    predictions_path = os.path.join(FLAGS.input_images, regex)
    predictions = glob.glob(predictions_path)
    leaf_count_acc = 0.0
    for path in predictions:
        prediction = np.load(path)
        unique, indices, counts = np.unique(prediction.reshape(-1, prediction.shape[2]), return_counts=True,
                                            return_index=True, axis=0)
        leaves_mask = (unique[:, 0] == 1)
        leaf_IDs = (unique[leaves_mask, :])
        leaf_IDs = leaf_IDs[:, 1]
        leaf_count_acc += float(leaf_IDs.shape[0]) / len(predictions)  # need to take away any stem or background values
    return leaf_count_acc

def process_side_rotations_images(x):
    regex = "side_Skeleton_" + str(x) + "*_mask.png"
    average = count_average_leaves_for_plant_images(regex)
    return average

def process_top_rotations_images(x):
    regex = "top_Skeleton_" + str(x) + "*_mask.png"
    average = count_average_leaves_for_plant_images(regex)
    return average

def count_average_leaves_for_plant_images(regex):
    predictions_path = os.path.join(FLAGS.input_images, regex)
    predictions = glob.glob(predictions_path)
    leaf_count_acc = 0.0
    for path in predictions:
        prediction = Image.open(path)
        prediction = np.asarray(prediction)
        # prediction = np.dot(prediction, [1, 256, 256 * 256])
        unique, indices, counts = np.unique(prediction.reshape(-1, prediction.shape[2]), return_counts=True,
                                            return_index=True, axis=0)
        # Convert to single unique value
        unique = np.dot(unique, [1, 256, 256 * 256])
        unique = unique[(unique != stem) & (unique != background)]
        leaf_count_acc += float(unique.shape[0]) / len(predictions)
    return leaf_count_acc


def main(_):

    print("Side Rotations")

    # We have images of plants numbered 5000-5099
    side_list = None
    with parallel_backend('multiprocessing'):
        side_list = Parallel(n_jobs=16)(delayed(process_side_rotations)(i) for i in range(5000, 5100))
        x = 5000
        for count in side_list:
            print("\tAverage leaf count for plant", x, "over rotation:", count)
            x += 1

    print("Top Rotations")
    # We have images of plants numbered 5000-5099
    top_list = None
    with parallel_backend('multiprocessing'):
        top_list = Parallel(n_jobs=16)(delayed(process_top_rotations)(i) for i in range(5000, 5100))
        x = 5000
        for count in top_list:
            print("\tAverage leaf count for plant", x, "over rotation:", count)
            x += 1

    # Let's make a csv file output
    plantNums = list(range(5000, 5100))
    dictTmp = {'plantNum': plantNums, 'leavesSide': side_list, 'leavesTop': top_list}
    df = pd.DataFrame(dictTmp)
    df.to_csv(FLAGS.output_csv_name)


if __name__ == '__main__':
  app.run(main)
