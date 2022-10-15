from audioop import add
from gc import callbacks
from multiprocessing import pool
from pyexpat import model
from tkinter import W
import cv2
import pandas as pd
import numpy as np
import imageio
import tqdm
import tensorflow as tf
import shutil
import tensorflow.keras.layers as kl
import keras_tuner
import matplotlib.pyplot as plt
import math
from box import Box
from keras.utils.layer_utils import count_params
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from toolz.curried import *

# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def save_gif(images, fps):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=fps)
    # return embed.embed_file("animation.gif")

def get_timmy_frames(total_n, num_frames):
    timmy_frames = []
    cap = cv2.VideoCapture('test_timmy_vod.ts')
    for i in range(total_n):
        ret, frame = cap.read()
        if i>(total_n-num_frames):
            timmy_frames.append(frame[:,:,[2,1,0]])
    cap.release()
    return np.array(timmy_frames)

def luminosity_grayscale(frame):
    return 0.0722 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.216 * frame[..., 2]

def read_avi(path, seq_frames, max_frames=99999):
    # READING INPUT
    cap = cv2.VideoCapture(path)
    full_vid = []
    try:
        for i in tqdm.tqdm(range(min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames))):
            ret, frame = cap.read()
            if not ret:
                break
            # opencv returns BGR format by default so we flip the luminosity formula weights to turn to greyscale
            full_vid.append((luminosity_grayscale(frame))[..., None].astype('uint8'))
    finally:
        cap.release()
    cut_frames = len(full_vid) - len(full_vid) % seq_frames
    return np.array(np.split(np.array(full_vid)[:cut_frames,:,:], cut_frames/seq_frames))

def frames_to_time(frames):
    secs = frames/2
    min_total = int(secs/60)
    sec_total = int(secs%60)
    return f"{min_total:0>2}:{sec_total:0>2}"

def time_to_frames(min,secs):
    return (min*60 + secs) * 2

def create_labels(n_rows, frame_boundaries):
    label_out = np.zeros(shape=(n_rows,1), dtype='bool')
    for x in frame_boundaries:
        label_out[x[0]:x[1],0] = True
    return label_out

def define_model(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    d1 = tf.keras.layers.Flatten()(d_in)
    d2 = tf.keras.layers.Dense(64, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(1, activation="sigmoid")(d2)
    return tf.keras.Model(inputs = d_in, outputs = d3, name='dense_model')

def define_lstm_model(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.ConvLSTM2D(2, 3, 2, return_sequences=True, data_format="channels_last")(d_in)
    x = tf.keras.layers.ConvLSTM2D(2, 3, 2, return_sequences=True, data_format="channels_last")(x)
    x = tf.keras.layers.ConvLSTM2D(2, 3, 2, return_sequences=True, data_format="channels_last")(x)
    x = tf.keras.layers.ConvLSTM2D(2, 3, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(4, 1, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(3, 3, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(3, 3, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(3, 3, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(3, 3, 2, return_sequences=True, data_format="channels_last")(x)
    # x = tf.keras.layers.ConvLSTM2D(4, 4, 4, return_sequences=True, data_format="channels_last")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='gru_model')

def define_simple_model(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Resizing(36, 64))(d_in)
    x = kl.TimeDistributed(kl.Flatten())(x)
    x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='simple_model')

def define_simple_cnn(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(d_in)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_simple_cnn_gru(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(d_in)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.GRU(16, return_sequences=True)(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_simple_cnn_gru_dropout(in_shape):
    d_in = tf.keras.Input(shape=in_shape)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(d_in)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.GRU(16, return_sequences=True)(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_simple_cnn_gru_dropout_reg(in_shape):
    reg = tf.keras.regularizers.l2(0.0001)
    d_in = tf.keras.Input(shape=in_shape)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(d_in)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.GRU(16, return_sequences=True, kernel_regularizer=reg)(x)
    x = kl.TimeDistributed(kl.Dropout(0.5))(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_complex_cnn_gru(in_shape, add_dropout=None):
    reg = tf.keras.regularizers.l2(0.0001)
    d_in = tf.keras.Input(shape=in_shape)
    def add_dropout_l(x):
        if add_dropout is not None:
            return kl.TimeDistributed(kl.Dropout(add_dropout))(x)
        else:
            return x
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(d_in)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.GRU(16, return_sequences=True, kernel_regularizer=reg)(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_complex_cnn_gru_bi(in_shape, add_dropout=None):
    reg = tf.keras.regularizers.l2(0.0001)
    d_in = tf.keras.Input(shape=in_shape)
    def add_dropout_l(x):
        if add_dropout is not None:
            return kl.TimeDistributed(kl.Dropout(add_dropout))(x)
        else:
            return x
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(2, 3, activation='relu', kernel_regularizer=reg))(d_in)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 3, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.Bidirectional(kl.GRU(16, return_sequences=True, kernel_regularizer=reg))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def define_complex_cnn_gru_bi_v2(in_shape, add_dropout=None, reg_val=0.0001):
    reg = tf.keras.regularizers.l2(reg_val)
    d_in = tf.keras.Input(shape=in_shape)
    def add_dropout_l(x):
        if add_dropout is not None:
            return kl.TimeDistributed(kl.Dropout(add_dropout))(x)
        else:
            return x
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu", return_sequences=True))(d_in)
    x = kl.TimeDistributed(kl.Conv2D(8, 5, 2, activation='relu', kernel_regularizer=reg))(d_in)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 5, 2, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Conv2D(8, 5, 2, activation='relu', kernel_regularizer=reg))(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(2))(x)
    x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Flatten())(x)
    # x = kl.TimeDistributed(kl.Dense(16, activation="relu"))(x)
    x = kl.Bidirectional(kl.GRU(8, return_sequences=True, kernel_regularizer=reg))(x)
    # x = add_dropout_l(x)
    x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"))(x)
    return tf.keras.Model(inputs = d_in, outputs = x, name='cnn')

def add_dropout_l(x, prm, hp = None, name=None):
    if hp is not None:
        prm.dropout_f = hp.Boolean('dropout_f')
        prm.dropout_val = hp.Float('dropout_val', 0.000001, 0.01, sampling='log')
    
    if prm.dropout_f:
        return kl.TimeDistributed(kl.Dropout(prm.dropout_val), name=name)(x)
    else:
        return x

def get_reg(prm, hp=None, name_mod=''):
    if hp is not None:
        prm.reg_f = hp.Boolean(f'{name_mod}reg_f')
        prm.reg_val = hp.Float(f'{name_mod}reg_val', 0.000001, 1e-5, sampling='log')
    
    if prm.reg_f:
        return tf.keras.regularizers.l2(prm.reg_val)

def add_conv_pipeline(x, i, prm, hp=None):
    for ii in range(prm.n_conv):
        x = kl.TimeDistributed(kl.Conv2D(
          **prm.conv_args,
        #   activation='relu', # hmmm
          kernel_regularizer=get_reg(prm, hp)),
          name=f'{i}_{ii}_conv')(x)
    x = kl.TimeDistributed(kl.MaxPooling2D(**prm.pool_args), name=f'{i}_{ii+1}_maxpool')(x)
    # x = add_dropout_l(x, prm, hp, name=f'{i}_{ii+2}_dropout')
    return x

class cnn_conv_lstm():
    def __init__(self, in_shape, **kwargs):
        self.in_shape = in_shape
        self.init_kwargs = kwargs
    
    def param(self, **kwargs):
        return Box(conv_lstm_filters=8)
    
    def model(self, hp=None):
        prm = self.param()
        d_in = tf.keras.Input(self.in_shape)
        x = d_in
        
        x = kl.TimeDistributed(kl.Conv2D(
          16, 11, 4,
          padding='same',
          kernel_regularizer=get_reg(prm, hp)),
          name=f'1_in_conv')(x)
        x = kl.LayerNormalization(name='1_layer_norm')(x)
        x = kl.Bidirectional(kl.ConvLSTM2D(
            4, 8, 2, 
            padding='same', return_sequences=True,
            kernel_regularizer=get_reg(prm, hp)),
            name='2_convlstm')(x)
        x = kl.LayerNormalization(name='2_layer_norm')(x)
        x = kl.Bidirectional(kl.ConvLSTM2D(
            4, 5, 2,
            padding='same', return_sequences=True,
            kernel_regularizer=get_reg(prm, hp)),
            name='3_convlstm')(x)
        x = kl.LayerNormalization(name='3_layer_norm')(x)
        x = kl.Bidirectional(kl.ConvLSTM2D(
            4, 5, 2,
            padding='same', return_sequences=True,
            kernel_regularizer=get_reg(prm, hp)),
            name='4_convlstm')(x)
        x = kl.LayerNormalization(name='4_layer_norm')(x)
        x = kl.TimeDistributed(kl.Conv2D(
          1, (12,20),
          activation='sigmoid',
          kernel_regularizer=get_reg(prm, hp)),
          name=f'5_out_conv')(x)
        return cmp_mod(tf.keras.Model(inputs = d_in, outputs = x, name='conv_lstm2d'), prm, hp)
        # x = kl.Bidirectional(kl.GRU(prm.gru_dim,
        #     return_sequences=True,
        #     kernel_regularizer=get_reg(prm, hp)), name=f'9_bigru')(x)


class cnn_bi_gru_small():
    def __init__(self, in_shape, **kwargs):
        self.in_shape = in_shape
        self.init_kwargs = kwargs

    def param(self, **kwargs):
        return Box(
            n_conv=1,
            conv_args=Box(filters=8, kernel_size=5, strides=2),
            pool_args=Box(pool_size=2),
            reg_val=None,
            dropout=None,
            default_box=True,
            default_box_attr=None,
            gru_dim = 10,
            learn_rate=0.0001
            ) | Box(**kwargs) | Box(**self.init_kwargs)
    
    def model(self, hp=None, **kwargs):
        prm = self.param(**kwargs)
        if hp is not None:
            prm.gru_dim = hp.Int('gru_dim', 250, 350)
        d_in = tf.keras.Input(self.in_shape)
        x = d_in
        # x = kl.TimeDistributed(kl.BatchNormalization(), name='1_input_norm')(x)
        for i in range(3):
            x = add_conv_pipeline(x, i, self.param(), hp)
            # x = kl.LayerNormalization(name=f'{i}_layer_norm')(x)
            x = add_dropout_l(x, prm, hp, name=f'{i}_dropout')
        # x = kl.TimeDistributed(kl.BatchNormalization(), name='3_middle_batch_norm')(x)
        # x = add_conv_pipeline(x, 4, self.param(n_conv=1,
        #  conv_args=Box(filters=8, kernel_size=3, strides=1)), hp)
        # x = kl.LayerNormalization(name='5_layer_norm')(x)
        # x = add_dropout_l(x, prm, hp, name=f'6_dropout')
        # x = add_conv_pipeline(x, 7, self.param(n_conv=1,
        #  conv_args=Box(filters=4, kernel_size=3, strides=1)), hp)
        # x = kl.LayerNormalization(name='8_layer_norm')(x)
        # x = add_dropout_l(x, prm, hp, name=f'9_dropout')
        # x = kl.LayerNormalization(name='5_layer_norm')(x)
        x = kl.TimeDistributed(kl.Flatten(), name=f'10_flatten')(x)
        x = add_dropout_l(x, prm, hp, name=f'11_dropout')
        x = kl.LayerNormalization(name='12_layer_norm')(x)
        x = kl.Bidirectional(kl.GRU(prm.gru_dim,
            return_sequences=True,
            kernel_regularizer=get_reg(prm, hp),
            recurrent_regularizer=get_reg(prm, hp, name_mod='rcrr_')
            ), name=f'9_bigru')(x)
        x = kl.TimeDistributed(kl.Dense(1, activation="sigmoid"), name=f'13_output')(x)
        return cmp_mod(tf.keras.Model(inputs = d_in, outputs = x, name='cnn_bi_gru_small'), prm, hp)

def compile_model(model_in, optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"]):
    return model_in.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

def get_optimizer(prm, hp=None):
    if hp is not None:
        prm.learn_rate = hp.Float('learn_rate', 0.00005, 0.001, sampling='log')
        prm.beta_1 = hp.Float('beta_1', 0.4, 0.99, sampling='log')
        prm.beta_2 = hp.Float('beta_2', 0.88, 0.999, sampling='log')

    return tf.keras.optimizers.Adam(learning_rate=prm.learn_rate, beta_1=prm.beta_1, beta_2=prm.beta_2)

def cmp_mod(model_in, prm, hp=None):
    model_in.compile(optimizer=get_optimizer(prm, hp), loss='binary_crossentropy')
    return model_in

def get_callbacks(run_param):
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = run_param.log_dir, profile_batch=run_param.profile_batch)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', **run_param.lr)
    stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=run_param.patience)
    return [tboard_callback, stopper]

def fit_mod(model_in, train_dataset, val_dataset, run_param):
    return model_in.fit(
        train_dataset,
        validation_data=val_dataset, 
        epochs=run_param.epoch, 
        callbacks=get_callbacks(run_param))

def tune_mod(model_f, train_dataset, val_dataset, run_param):
    tuner = keras_tuner.Hyperband(
        hypermodel=model_f,
        objective='val_loss',
        max_epochs=run_param.epoch,
        # executions_per_trial=run_param.exec_per_trial,
        overwrite=True,
        hyperparameters=run_param.hp,
        directory=run_param.log_dir,
        project_name='stream_labeller_test'
    )
    tuner.search(train_dataset, validation_data=val_dataset, epochs=run_param.epoch, callbacks=get_callbacks(run_param))
    return tuner

def compile_and_fit_model(model_in,
         train_dataset,
         val_dataset,
         run_param,
         optimizer=None,
         loss='binary_crossentropy'
         ):
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = 'training_logs', histogram_freq = 1, profile_batch = '500,520')
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', **run_param.lr)
    stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=run_param.patience)
    if optimizer is None:
        optimizer = get_optimizer()
    model_in.compile(optimizer=optimizer, loss=loss)
    history = model_in.fit(train_dataset, validation_data=val_dataset, epochs=run_param.epoch, callbacks=[tboard_callback, stopper, reduce_lr])
    print('done!')
    return history

def summarise_labels(label, offset):
    dat = pd.DataFrame([label, (label!=np.roll(label, 1)).cumsum()]).T
    dat.columns=['label', 'group']
    summary=dat.groupby('group').agg(
        label = ('label', 'first'), 
        start_index= ('label', lambda x: x.index.min()),
        end_index = ('label', lambda x: x.index.max()),
        n_frames = ('label', 'size'),
    ).reset_index()
    summary['adj_start'] = summary['start_index'] + offset
    summary['adj_end'] = summary['end_index'] + offset
    summary['adj_ts'] = summary['adj_start'].apply(p_frames_to_twitch_ts, args=(2,))
    summary['adj_end_ts'] = summary['adj_end'].apply(p_frames_to_twitch_ts, args=(2,))
    summary['duration'] = summary['n_frames'].apply(p_frames_to_twitch_ts, args=(2,))
    return summary.drop(columns=['adj_start', 'adj_end'])

def p_frames_to_twitch_ts(frame, fps):
    frame_s = frame/fps
    return f"{math.floor(frame_s/60/60)}h{math.floor((frame_s/60) % 60)}m{math.floor(frame_s % 60)}s"

def p_ts_to_int(ts):
    return np.array(list(map(int, ts.split(':'))))

def p_ts_to_frames(ts, fps):
    ts_int_to_sec = np.array([60*60, 60, 1])
    return (p_ts_to_int(ts)*ts_int_to_sec).sum()*fps

def p_ts_range_to_ints(ts_r):
    return list(map(p_ts_to_int, ts_r.split('-')))

def p_labels_to_frames(labels, fps):
    ranges = list(map(p_ts_range_to_ints, labels))
    ts_int_to_sec = np.array([60*60, 60, 1])
    return (np.array(ranges) * ts_int_to_sec).sum(axis=2)*fps

def create_label_array(label_ts, fps, n):
    frame_ranges = p_labels_to_frames(label_ts, fps)
    labels = np.zeros(n, dtype='bool')
    
    for s,f in frame_ranges:
        labels[s:f] = True
    return labels

def create_part_label_array(label_ts, fps, start, n):
  frame_ranges = p_labels_to_frames(label_ts, fps)
  filtered_labels = pipe(
    frame_ranges,
    filter(lambda x: not (x[1]<start or x[0]>(start+n))),
    map(lambda x: x-start)
  )
  labels = np.zeros(n, dtype='bool')
  
  for s,f in filtered_labels:
      labels[s:f] = True
  return labels

def np_gco(df): return np.unique(df, return_counts=True)

def record_metrics():
    shutil.rmtree('training_logs')
    return tf.keras.callbacks.TensorBoard(
        log_dir="training_logs",
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="epoch",
    )

def pred_vs_label(test_pred_flat, test_lab_nulled=None):
  test_idx = np.indices(test_pred_flat.shape).flatten()
  plt.figure(figsize=(12,8))
  test_pred_colors =  np.where(test_pred_flat>0.5, 'r', 'b')
  plt.scatter(test_idx, test_pred_flat, color=test_pred_colors)
  test_pred_binary =  np.where(test_pred_flat>0.5, 1.015, None)
  plt.scatter(test_idx, test_pred_binary, marker='s', s=1, color='red', label='predicted')
  if test_lab_nulled is not None:
    plt.scatter(test_idx, test_lab_nulled, marker='s', s=1, color='green', label='labels')
  plt.legend()
  plt.show()
  plt.close()

def plot_hist(hist):
    plot_multi_hist({'model': hist})

def plot_multi_hist(all_hist):
  plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
  plt.figure(figsize=(12,8))
  plotter.plot(all_hist)
  plt.show()
  plt.close()

def seq_batch_data(ds, run_param):
    return ds.batch(run_param.seq, drop_remainder=True).batch(run_param.batch)