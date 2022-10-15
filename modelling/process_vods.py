import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

import util as u
import importlib
importlib.reload(u)
import util as u
import skvideo.io

SEQ_FRAMES = 40
MAX_FRAMES = 10000

# read in avi
model_in = u.read_avi('timmy_model_apex.avi', SEQ_FRAMES, MAX_FRAMES)

# READING INPUT
cap = cv2.VideoCapture('timmy_model_apex.avi')
full_vid = []
try:
    for i in tqdm.tqdm(range(min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames))):
        ret, frame = cap.read()
        if not ret:
            break
        # opencv returns BGR format by default so we flip the luminosity formula weights to turn to greyscale
        full_vid.append((0.0722 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.216 * frame[..., 2])[..., None].astype('uint8'))
finally:
    cap.release()
cut_frames = len(full_vid) - len(full_vid) % seq_frames
np.array(np.split(np.array(full_vid)[:cut_frames,:,:], cut_frames/seq_frames))

ret, frame = cap.read()


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


u.luminosity_grayscale(frame).dtype
frame_feature = {
    'frame' : _bytes_feature(u.luminosity_grayscale(frame).tobytes()),
    'label' : _int64_feature(0)
}
single_record = tf.train.Example(features=tf.train.Features(feature=frame_feature))

np.frombuffer(single_record.features.feature['frame'].bytes_list.value[0], dtype='float64')
single_record.features.feature['frame'].bytes_list.value[0]
single_record.features.feature['label'].int64_list.value[0]

range_ds = tf.data.Dataset.range(100000)
batches = range_ds.batch(10, drop_remainder=True)

for batch in batches.take(5):
  print(batch.numpy())

videodata = skvideo.io.vread('timmy_model_apex.avi', as_grey=True)

# import sys
# sys.getsizeof(videodata)/1024/1024/1024
# videodata.shape