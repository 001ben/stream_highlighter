# powershell commands:
# streamlink https://www.twitch.tv/videos/1582854814 360p30 --hls-segment-threads 8 -o download_360p_60fps.ts
# ffmpeg -i download_360p_60fps.ts -r 2 -an download_360p_2fps.avi
# ffmpeg -i timmy_valo_twitch_rivals.ts -an -filter:v format=gray -r 2 timmy_valo_twitch_rivals_gray.avi
# streamlink https://www.twitch.tv/videos/1582854814 360p30 --hls-segment-threads 8 -O | ffmpeg -i pipe:0 -c copy -r 2 -an download_360p_2fps_test.avi

# ffmpeg -i download_360p_60fps.ts -r 2 -an download_360p_2fps.avi

# ffmpeg -i .\download_360p_2fps.avi -f segment -segment_time 720 -reset_timestamps 1 -break_non_keyframes 1 segment_video/out%03d.avi

# === setup
# import keras_tuner
# import tensorflow_docs.plots

import os
import os.path
import pathlib
import re
from cProfile import label
from datetime import datetime
from functools import partial
from glob import glob

import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_io as tfio
from box import Box
from tensorflow import keras
from tensorflow.keras import mixed_precision
from toolz import compose
from toolz.curried import *
from toolz.functoolz import identity
from toolz.sandbox.core import unzip

import modelling.tf_memory_usage as tfmu

def reload_u():
  import importlib
  import modelling.util as u
  importlib.reload(u)
  import modelling.util as u
  return u

u = reload_u()

# import tensorflow as tf
# import numpy as np
# dd = tf.data.Dataset.from_tensor_slices((np.arange(20), np.arange(20)))
# list(dd.window(3, drop_remainder=True).flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(3), y.batch(3)))).as_numpy_iterator())
# list(dd.window(3, drop_remainder=True).flat_map(lambda x, y: (x.batch(3), y.batch(3))).as_numpy_iterator())

# === Constants
SEQ_FRAMES = 120
BATCH_SIZE = 6
NUM_SAMPLES = 10
N_FRAMES_PER_VIDEO = 1440

# === initialise memory growth an mixed precision
map_l = curry(compose_left(map, list))

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
fps = 2
mixed_precision.set_global_policy('mixed_float16')

# === read in video and labels
data_root = pathlib.Path('output/stream_downloads')

box_from_yaml = lambda x: Box.from_yaml(filename=x).labels
filename_to_int = compose_left(os.path.basename, lambda x: x.replace('out', '').replace('.avi', ''), int)

create_labelled_video_tuple = lambda x, y: (
 str(y),
 u.create_part_label_array(
    x,
    fps,
    filename_to_int(y)*N_FRAMES_PER_VIDEO,
    N_FRAMES_PER_VIDEO))

load_labels = compose_left(
  lambda x: list(x.glob('labels.yaml')),
  first,
  box_from_yaml)

complete_videos = [
  'output\\stream_downloads\\original_first_time_apex\\segment_video'
]

train_test_split = lambda x: (
  (os.path.dirname(x[0]) not in complete_videos) or
  int(re.search(r'out(\d+)', os.path.basename(x[0])).group(1))<12
)

labelled_video_tuples = pipe(
  data_root.glob('*/labels.yaml'),
  map_l(compose_left(os.path.dirname, pathlib.Path)),
  mapcat(lambda x: [create_labelled_video_tuple(load_labels(x), y) for y in x.glob('segment_video/*')]),
  filter(lambda x: len(np.unique(x[1]))>1 or os.path.dirname(x[0]) in complete_videos),
  # filter(lambda x: os.path.dirname(x[0]) in complete_videos), # just to test
  groupby(lambda x: 'train' if train_test_split(x) else 'test'),
  valmap(compose_left(unzip, map_l(list))),
)
pipe(labelled_video_tuples, valmap(lambda x: len(x[0]))) # train/test count

predict_on = ['playing_apex_with_elephante']

prediction_videos = pipe(
  data_root.glob('*/segment_video/*'),
  # map_l(compose_left(os.path.dirname, pathlib.Path)),
  groupby(lambda x: x.parent.parent),
  keyfilter(lambda x: x.name in predict_on),
  valmap(map_l(str))
)

# === Video processing pipeline functions
@tf.numpy_function()
def skvideo_read(x):
  # print(f"processing video {x.numpy().decode()}")
  return skvideo.io.vread(x.numpy().decode(), as_grey=True)

@tf.function()
def load_video_windowed(x, y):
  window_ds = (
    tf.data.Dataset.from_tensor_slices((tf.py_function(skvideo_read, inp=[x], Tout=tf.uint8), y))
    .window(SEQ_FRAMES, shift=int(SEQ_FRAMES/4))
  )
  return (
    window_ds
    .flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(SEQ_FRAMES, drop_remainder=True), y.batch(SEQ_FRAMES, drop_remainder=True))))
    .shuffle(48)
    .batch(BATCH_SIZE, drop_remainder=True)
    .take(3)
  )

@tf.function()
def load_video(x, y):
  # vid = tf.py_function(skvideo_read, inp=[x], Tout=tf.uint8)[:500]
  vid = tf.py_function(skvideo_read, inp=[x], Tout=tf.uint8)
  return (
    tf.data.Dataset
    .from_tensor_slices((vid, y[:len(vid)]))
    .batch(SEQ_FRAMES)
    .padded_batch(BATCH_SIZE, ([None, None, None, 1], [None]))
  )

@tf.function
def load_just_video(x):
  return (
    tf.data.Dataset.from_tensor_slices(tf.py_function(skvideo_read, inp=[x], Tout=tf.uint8))
    .batch(SEQ_FRAMES)
    .padded_batch(BATCH_SIZE, ([None, None, None, 1]))
  )

# labelled_video_tensor = tf.data.Dataset.from_tensor_slices(tuple(labelled_video_tuples))
# list(labelled_video_tensor.skip(96).take(1).as_numpy_iterator())
# try predicting on smaller batch?
# pipe(
#   labelled_video_tensor
#   .skip(96)
#   .take(1)
#   .flat_map(load_video)
#   .as_numpy_iterator(),
#   list,
#   # map_l(len),
#   # take(1),
#   # map_l(map_l(map_l(len))),
#   map_l(lambda x: x[0].shape),
#   list
# )

# bmodel.predict(labelled_video_tensor
#   .skip(96)
#   .take(1)
#   .flat_map(load_video))

# list(labelled_video_tensor.as_numpy_iterator())

# === setup train/test dataset pipelines
train_dataset = (
  # labelled_video_tensor.take_while(lambda x,y: tf.strings. x.)
  tf.data.Dataset.from_tensor_slices(tuple(labelled_video_tuples['train']))
  .shuffle(200)
  .interleave(load_video_windowed, cycle_length=6, block_length=2, num_parallel_calls=6)
  .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
  tf.data.Dataset.from_tensor_slices(tuple(labelled_video_tuples['test']))
  .flat_map(load_video)
  .cache()
)

labelled_video_tuples

# vid_path = 'output/stream_downloads/original_video/timmy_model_apex.avi'
# videodata = skvideo.io.vread(vid_path, as_grey=True)
# label_ts  = Box.from_yaml(filename='output/stream_downloads/original_first_time_apex/labels.yaml').labels
# labels = u.create_label_array(label_ts, fps, videodata.shape[0])
# u.p_labels_to_frames(label_ts, fps)

# === setup default run params
u = reload_u()

default_run_param = Box(
  epoch=100,
  patience=10,
  batch=BATCH_SIZE,
  seq=SEQ_FRAMES,
  lr=Box(factor=0.2, patience=5, min_lr=0.0001),
  log_dir="training_logs",
  max_trials=10,
  exec_per_trial=1,
  hp=None,
  default_box=True, default_box_attr=None,
  profile_batch=0)

# test_dataset = (
#   tf.data.Dataset.from_generator(gen_frames,
#    args=splits[1:], output_signature=shape_spec))
# test_batch = u.seq_batch_data(test_dataset, default_run_param).prefetch(tf.data.AUTOTUNE)

# === setup some runner config (might delete)
all_histories={}

def run_experiment(e_name, model_f, param_d=None, run_param=default_run_param):
  train_batch = train_dataset.batch(run_param.seq, drop_remainder=True).batch(run_param.batch)
  test_batch = test_dataset.batch(run_param.seq, drop_remainder=True).batch(run_param.batch)
  model = model_f(train_batch.element_spec[0].shape[1:], param_d)
  model.summary()
  all_histories[e_name] = u.compile_and_fit_model(model, train_batch, test_batch, run_param)
  plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
  plt.figure(figsize=(12,8))
  plotter.plot(all_histories)
  plt.show()

# === define model
in_shape = [SEQ_FRAMES, 360, 640, 1] # in_shape = [30, 360, 640, 1]

u = reload_u()
model_cls = u.cnn_bi_gru_small(in_shape)
# model_cls = u.cnn_conv_lstm(in_shape)

# === base model first
man_hp = keras_tuner.HyperParameters()
man_hp.Fixed('learn_rate', 0.00062617)
man_hp.Fixed('beta_1', 0.71384)
man_hp.Fixed('beta_2', 0.98336)
man_hp.Fixed('dropout_f', False)
man_hp.Fixed('dropout_val', 0.02)
man_hp.Fixed('reg_f', False)
man_hp.Fixed('reg_val', 0.00001)
man_hp.Fixed('rcrr_reg_f', False)
man_hp.Fixed('rcrr_reg_val', 0.00001)
man_hp.Fixed('gru_dim', 150)

bmodel = model_cls.model(man_hp)
bmodel.summary()
man_hp.space

tfmu.print_memory_usage(bmodel, BATCH_SIZE)

def get_curr_time():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

train_hist = u.fit_mod(bmodel, train_dataset, test_dataset, run_param = default_run_param | Box(
  patience=6,
  epoch=6,
  log_dir=f"tuning/new_pipeline_test/run_{get_curr_time()}"))

u.plot_hist(train_hist)

# === tune the model
fix_hp = keras_tuner.HyperParameters()
fix_hp.Fixed('dropout_f', False)
fix_hp.Fixed('reg_f', True)
fix_hp.Fixed('rcrr_reg_f', True)
fix_hp.Fixed('dropout_val', 0.02)
fix_hp.Fixed('gru_dim', 250)
# fix_hp.Fixed('rcrr_reg_val', 0.00001)

fix_hp.space

tune_params = default_run_param | Box(
  log_dir="tuning/tuning_largedataset_8hrs_labelled",
  max_trials=3,
  exec_per_trial=70,
  patience=7,
  epoch=55,
  hp=fix_hp)

tuner = u.tune_mod(model_cls.model, train_dataset, test_dataset, tune_params)

tuner.search_space_summary()
dir(tuner)
tuner.results_summary(1)


best_hps = tuner.get_best_hyperparameters(1)
best_hps[0].values
abc = tuner.get_best_models(1)
abc[0].evaluate(test_dataset)
# Build the model with the best hp.
chosen_hp = best_hps[1]
chosen_hp.values
tuned_model = model_cls.model(chosen_hp)

train_hist = u.fit_mod(tuned_model, train_batch, test_batch, default_run_param | Box(
  patience=15,
  epoch=70,
  log_dir=f"tuning/tuned_models/run_{get_curr_time()}"))

u.plot_hist(train_hist)


# === sorta legacy
all_histories = {}
all_histories['comp_cnn_gru_bi_d02_l001'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.02, reg_val=0.001), train_batch, test_batch, 300, patience=40)
all_histories['comp_cnn_gru_bi_d04_l001'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.001), train_batch, test_batch, 300, patience=40)
all_histories['comp_cnn_gru_bi_d05_l001'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.05, reg_val=0.001), train_batch, test_batch, 300, patience=40)

# all_histories['simple'] = u.compile_and_fit_model(simple_model, train_batch, test_batch, 100)
# all_histories['cnn'] = u.compile_and_fit_model(u.define_simple_cnn(train_batch.element_spec[0].shape[1:]), train_batch, test_batch, 100)
# all_histories['cnn_gru'] = u.compile_and_fit_model(u.define_simple_cnn_gru(train_batch.element_spec[0].shape[1:]), train_batch, test_batch, 200, patience=20)
# all_histories['cnn_gru_do'] = u.compile_and_fit_model(u.define_simple_cnn_gru_dropout(train_batch.element_spec[0].shape[1:]), train_batch, test_batch, 200, patience=20)
# all_histories['cnn_gru_do_reg'] = u.compile_and_fit_model(u.define_simple_cnn_gru_dropout_reg(train_batch.element_spec[0].shape[1:]), train_batch, test_batch, 200, patience=20)

# all_histories['comp_cnn_gru'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:]), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d25'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.25), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d50'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.5), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d10'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.1), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d03'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.03), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d02'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.02), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_d01'] = u.compile_and_fit_model(u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.01), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d02'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi(train_batch.element_spec[0].shape[1:], add_dropout=0.02), train_batch, test_batch, 300, patience=30)

# all_histories['comp_cnn_gru_bi_d02_v2'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.02), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d20_v2'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.2), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d02_v2_retry'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.02), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d04_v2'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d04_l01'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.01), train_batch, test_batch, 300, patience=30)


# all_histories['comp_cnn_gru_bi_d04_l001_v2_batch16'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.001), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d04_l001_v2_batch32'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.001), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d04_l005_v2'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.005), train_batch, test_batch, 300, patience=30)
# all_histories['comp_cnn_gru_bi_d06_l001_v2'] = u.compile_and_fit_model(u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.06, reg_val=0.001), train_batch, test_batch, 300, patience=30)

plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')

plt.figure(figsize=(12,8))
# plotter.plot({k:v for k,v in all_histories.items() if k in ['cnn', 'cnn_gru_do'] or k.startswith('comp_')})
# plt.title('cnn_gru models with 1h 15min labelled data')
plotter.plot(all_histories)
# plt.savefig('history_plot.png')
plt.close()

plt.figure(figsize=(12,8))
plotter.plot({k:v for k,v in all_histories.items() 
  if k in ['comp_cnn_gru_d10', 'comp_cnn_gru', 'comp_cnn_gru_d02', 'comp_cnn_gru_d01', 'comp_cnn_gru_d03']})
plt.close()

# s_model = u.define_complex_cnn_gru_bi_v2(train_batch.element_spec[0].shape[1:], add_dropout=0.04, reg_val=0.001)
# history = u.compile_and_fit_model(s_model, train_batch, test_batch, 85, patience=60)

# s_model = u.define_complex_cnn_gru(train_batch.element_spec[0].shape[1:], add_dropout=0.02)
s_model = abc[0]

s_model = bmodel
s_model.summary()
u.compile_model(s_model, metrics=['accuracy'])
s_model.evaluate(test_dataset)

s_model.save('output/models/cnn_bi_gru_294_04_10_2022')

[[y.shape for y in x.layer.weights] for x in s_model.layers[1:8]]

res = list(test_dataset.take(1).as_numpy_iterator())[0]

inp_image = res[0][0]
dir(s_model.layers[1].layer)

k_in = keras.Input(shape=inp_image.shape[1:])
conv_tree = reduce(lambda x,y: y(x), [x.layer for x in s_model.layers[1:8]], k_in)

test_model = keras.Model(
  inputs = k_in,
  outputs = conv_tree)
test_model.predict(inp_image).shape

inp_image.shape
360*640


keras.backend.function([keras.Input(shape=inp_image.shape)], [s_model.layers[1].layer.output])

# s_model = tf.keras.models.load_model('output/models/cnn_bi_gru_150_03_10_2022')

plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
plt.figure(figsize=(12,8))
plotter.plot({'model': history})

test_preds = s_model.predict(test_dataset)
test_pred_flat = test_preds.flatten()
# test_summary = u.summarise_labels(frame_smooth, splits[2])
# test_preds.flatten().round()

# === Test Labels
test_lab_out = np.concatenate([x.numpy() for x in test_dataset.unbatch().map(lambda x,y: y)])
# test_lab_out = np.array([x.numpy() for x in test_batch.flat_map(lambda x,y: y)])
test_lab_nulled = np.where(~test_lab_out, None, test_lab_out)

# === Test plots
u.pred_vs_label(test_pred_flat, test_lab_nulled)
u.pred_vs_label(pd.Series(test_pred_flat).rolling(14, center=True, min_periods=1).mean(), test_lab_nulled)
u.pred_vs_label(pd.Series(test_pred_flat).rolling(10, min_periods=1).mean(), test_lab_nulled)

# === Test Summary
test_summary = u.summarise_labels(pd.Series(test_pred_flat).rolling(10, center=True, min_periods=1).mean().round(), splits[1])
test_summary.query('label==1').reset_index(drop=True)
# u.frames_to_time(splits[2])

plt.close('all')

# === Predict more frames

for k,v in prediction_videos.items():
  print(v)


prediction_dataset = (
  tf.data.Dataset.from_tensor_slices(v)
  .flat_map(load_just_video)
)

predictions = s_model.predict(prediction_dataset)
frame_smooth = pd.Series(predictions.flatten()).rolling(10, center=True, min_periods=1).mean()
frame_smooth[1::2].to_json(k / 'predictions.json', orient='records')
label_summarise = u.summarise_labels(frame_smooth.round(), 0)
export_cols = np.floor(label_summarise.query("label==1").reset_index(drop=True)[['start_index', 'end_index']]/2).astype('int').reset_index()
export_cols.columns = ['id', 'start', 'end']
export_cols.to_json(k / 'timestamps.json', orient='records')

# {{{

def predict_video(vid_path, out_folder, s_model):
  predvideodata = skvideo.io.vread(vid_path, as_grey=True)
  os.makedirs(out_folder, exist_ok=True)
  
  # === setup splits and datasets
  def pred_gen_frames():
    i = 0
    while i < predvideodata.shape[0]:
      yield predvideodata[i]
      i+=1
  
  pred_shape_spec = (
    tf.TensorSpec(shape=predvideodata.shape[1:], dtype=tf.uint8)
  )
   
  pred_dataset = (
    tf.data.Dataset.from_generator(pred_gen_frames, output_signature=pred_shape_spec))
  SEQ_FRAMES = 120
  BATCH_SIZE = 12
  pred_batch = (
    u.seq_batch_data(pred_dataset, Box(seq=SEQ_FRAMES, batch=BATCH_SIZE))
    .prefetch(tf.data.AUTOTUNE)
  )
  predictions = s_model.predict(pred_batch)
  frame_smooth = pd.Series(predictions.flatten()).rolling(10, center=True, min_periods=1).mean()
  frame_smooth[1::2].to_json(os.path.join(out_folder,'predictions.json'), orient='records')
  label_summarise = u.summarise_labels(frame_smooth.round(), 0)
  export_cols = np.floor(label_summarise.query("label==1").reset_index(drop=True)[['start_index', 'end_index']]/2).astype('int').reset_index()
  export_cols.columns = ['id', 'start', 'end']
  # export_cols.reset_index(name='id')
  export_cols.to_json(os.path.join(out_folder, 'timestamps.json'), orient='records')

s_model = keras.models.load_model('output/models/cnn_bi_gru_200_25_09_2022')

predict_video(
 'output/stream_downloads/timmy_valo_twitch_rivals/timmy_valo_twitch_rivals.avi',
 'output/stream_downloads/timmy_valo_twitch_rivals_exp2/',
 s_model
)

predict_video(
 'output/stream_downloads/original_video/timmy_model_apex.avi',
 'output/stream_downloads/original_video/',
 s_model
)

with tf.device('/CPU:0'):
  predict_video(
  'output/stream_downloads/playing_apex_with_elephante/download_360p_2fps.avi',
  'output/stream_downloads/playing_apex_with_elephante/',
  s_model
  )

# === Interrogating predictions
predvideodata = skvideo.io.vread( 'output/stream_downloads/original_video/timmy_model_apex.avi', as_grey=True)
pp= pd.read_json('output/stream_downloads/original_video/predictions.json')
u = reload_u()
u.p_ts_to_int('03:37:52')

pp.iloc[u.p_ts_to_frames('03:40:02', 1)]
predvideodata[u.p_ts_to_frames('03:40:02', 1)]

plt.figure(figsize=(10, 10))
plt.imshow(predvideodata[u.p_ts_to_frames('03:40:05', 2)], cmap='gray')
plt.axis("off")


# === Setup sequence length and batch size

# }}}

with tf.device('/cpu:0'):
# with tf.device('/gpu:0'):
  pred_ds = tf.data.Dataset.from_tensor_slices(videodata)
  pred_batch = pred_ds.batch(SEQ_FRAMES, drop_remainder=True).batch(BATCH_SIZE*2)
  predictions = s_model.predict(pred_batch)

frame_smooth = pd.Series(predictions.flatten()).rolling(10, center=True, min_periods=1).mean()
frame_smooth[1::2].to_json('data/timmy_model_apex_preds.json', orient='records')

# u.p_frames_to_twitch_ts(2000, 2)

label_summarise = u.summarise_labels(frame_smooth.round(), 0)
# pd.options.display.chop_threshold=30
# label_summarise.query('label==1')
# label_summarise['start_link'] = '<a href="https://www.twitch.tv/videos/1577547277?t=' + label_summarise['adj_ts'] + '">' + label_summarise['adj_ts'] + '</a>'

export_cols = np.floor(label_summarise.query("label==1").reset_index(drop=True)[['start_index', 'end_index']]/2).astype('int').reset_index()
export_cols.columns = ['id', 'start', 'end']
# export_cols.reset_index(name='id')
export_cols.to_json('data/timmy_model_apex_pred_frames.json', orient='records')

# display.HTML(label_summarise.query('label==1')[:40].to_html(escape=False))
# u.frames_to_time(10000)

# tf.config.experimental.get_memory_info('GPU:0')['current']/1024/1024/1024

# %load_ext tensorboard
# %tensorboard --logdir training_logs --host localhost --port 8888

# import matplotlib.pyplot as plt
# plt.plot([])
