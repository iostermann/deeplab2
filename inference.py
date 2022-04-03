import collections
import os
import tempfile

from absl import app
from absl import flags
from absl import logging

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import urllib

import tensorflow as tf

from deeplab2 import common
from deeplab2.trainer import vis
from deeplab2.trainer import vis_utils


flags.DEFINE_string(
    'trained_model',
    default=None,
    help='The path to a saved pretrained model')

flags.DEFINE_string(
  'input_image',
  default=None,
  help='image to run inference on'
)

FLAGS = flags.FLAGS

def main(_):
  model_name = FLAGS.trained_model
  loaded_model = tf.saved_model.load(model_name)

  with tf.io.gfile.GFile(FLAGS.input_image, 'rb') as f:
    im = np.array(Image.open(f).convert('RGB'))

  predictions = loaded_model(im)
  predictions = {key: predictions[key][0] for key in predictions}
  #predictions = vis_utils.squeeze_batch_dim_and_convert_to_numpy(predictions)
  if common.PRED_INSTANCE_KEY in predictions:
      min_instance_pred = np.min(predictions[common.PRED_INSTANCE_KEY])
      max_instance_pred = np.max(predictions[common.PRED_INSTANCE_KEY])
      rgb = vis_utils.create_rgb_from_instance_map(predictions[common.PRED_INSTANCE_KEY])

  print(predictions)

if __name__ == '__main__':
  app.run(main)
