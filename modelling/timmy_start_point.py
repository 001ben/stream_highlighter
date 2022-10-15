# ffmpeg -i test_timmy_vod.ts -filter:v fps=2 -to 04:24:00 -an timmy_model_apex.avi
# 
# !pip install tqdm
# !pip install --upgrade pandas

import cv2
import pandas as pd
import numpy as np
import imageio
import tqdm
import tensorflow as tf
import shutil
from PIL import Image

import util as u
import importlib
importlib.reload(u)
import util as u

SEQ_FRAMES = 40
MAX_FRAMES = 10000

# read in avi
model_in = u.read_avi('timmy_model_apex.avi', SEQ_FRAMES, MAX_FRAMES)
# model_in = model_in[..., None].copy()
# model_in.shape

# test show an image
# Image.fromarray(model_in[50][0].astype('uint8'))

# t_frame = frames_gray.shape[0] - frames_gray.shape[0]%SEQ_FRAMES
# N_SPLITS = t_frame/SEQ_FRAMES

# model_in = np.array(np.split(frames_gray[:t_frame,:,:100], N_SPLITS))
# model_in.shape

# u.time_to_frames(4, 36)
# u.frames_to_time(350)
label_frames = [
    (u.time_to_frames(22, 55), u.time_to_frames(24, 36)),
    (u.time_to_frames(26, 35), u.time_to_frames(27, 26)),
    (u.time_to_frames(29, 33), u.time_to_frames(30, 29)),
    (u.time_to_frames(30, 40), u.time_to_frames(31, 20)),
    (u.time_to_frames(32, 30), u.time_to_frames(33, 10)),
    (u.time_to_frames(33, 45), u.time_to_frames(34, 00)),
    (u.time_to_frames(34, 13), u.time_to_frames(36, 33)),
    (u.time_to_frames(39, 40), u.time_to_frames(40, 20)),
    (u.time_to_frames(43, 32), u.time_to_frames(43, 55)),
    (u.time_to_frames(44, 13), u.time_to_frames(45, 4)),
    (u.time_to_frames(45, 40), u.time_to_frames(46, 37)),
    (u.time_to_frames(47, 58), u.time_to_frames(48, 40)),
]
labels = np.array(np.split(u.create_labels(model_in.shape[0]*model_in.shape[1], label_frames), model_in.shape[0]))

first_label_frame_split = int(np.ceil(label_frames[0][0]/SEQ_FRAMES)) - 2
last_label_frame_split = int(np.ceil(label_frames[-1][1]/SEQ_FRAMES))

# train_idx = np.random.choice(np.indices(frames_gray.shape[:1])[0], size=int(frames_gray.shape[0]*0.70))
# train_idx_sub = np.zeros(shape=frames_gray.shape[0], dtype='bool')
# train_idx_sub[train_idx] = True

spl = (first_label_frame_split,int(first_label_frame_split + (last_label_frame_split-first_label_frame_split)*0.75), last_label_frame_split)

train_x = model_in[spl[0]:spl[1], ...]
train_y = labels[spl[0]:spl[1], ...]
test_x = model_in[spl[1]:spl[2], ...]
test_y = labels[spl[1]:spl[2], ...]

# np.unique(train_y, return_counts=True)
# np.unique(test_y, return_counts=True)


importlib.reload(u)
import util as u
model = u.define_lstm_model(train_x.shape[1:])
model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

shutil.rmtree('training_logs')
metrics_callback = tf.keras.callbacks.TensorBoard(
    log_dir="training_logs",
    histogram_freq=0,  # How often to log histogram visualizations
    embeddings_freq=0,  # How often to log embedding visualizations
    update_freq="epoch",
)
train_hist = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=8, epochs=20, callbacks=[metrics_callback])
results = model.evaluate(test_x, test_y, batch_size=10)
# np.unique(test_y, return_counts=True)

# model_in.shape
# model_in[:spl[2]].shape[0] + model_in[spl[2]:].shape[0]

# investigate prediction frames
pred_idx = 106
u.frames_to_time((spl[2]) * SEQ_FRAMES)
Image.fromarray(model_in[spl[2]:][int(pred_idx/SEQ_FRAMES)][pred_idx % SEQ_FRAMES][..., 0])


rest_of_video = model.predict(model_in[spl[2]:])
# rest_of_video[:,:,0].tolist()
smoother_predicted_fights = pd.Series(rest_of_video.flatten()).rolling(12, min_periods=1).mean().round()

smooth_sum = u.summarise_labels(smoother_predicted_fights)
smooth_sum['adj_start'] = smooth_sum['start_index'] + (spl[2]) * SEQ_FRAMES
smooth_sum['adj_end'] = smooth_sum['end_index'] + (spl[2]) * SEQ_FRAMES
smooth_sum['adj_ts'] = smooth_sum['adj_start'].apply(u.frames_to_time)
smooth_sum['adj_end_ts'] = smooth_sum['adj_end'].apply(u.frames_to_time)
smooth_sum['duration'] = smooth_sum['n_frames'].apply(u.frames_to_time)
smooth_sum.query('label==1')

label_times = [tuple(u.frames_to_time(y) for y in x) for x in label_frames]

tuple(u.frames_to_time(y) for y in spl)
results
label_times
smooth_sum


np.unique(np.round(rest_of_video), return_counts=True)
np.unique(rest_of_video, return_counts=True)


pred_frames = model.predict(train_x)
u.summarise_labels(pred_frames)
np.unique(pred_frames, return_counts=True)



# SHOW A FRAME

WIN_X = "MYWIN"
def create_window(WIN_X):
    cv2.namedWindow(WIN_X, cv2.WINDOW_AUTOSIZE)

def show_frame(WIN_X, i):
    i = 350
    current_area = summarised_labels.query("(@i >= start_index) & (@i <= end_index)")
    next_area = summarised_labels.query("@i < start_index").head(1)
    next_area.get('start_index')

    def take_srs(srs, dflt=-1):
        return take_with_default(srs.tolist(), dflt)
    def take_with_default(lst, dflt=-1):
        return next(iter(lst), dflt)

    frame_to_display = timmy_vid_letsgo[i][:,:,[2,1,0]].copy()
    display_text = "Action" if label[0, i] else "Waiting"
    display_text += f" {i} ({take_srs(current_area.start_index)} - {take_srs(current_area.end_index)}) | next ({take_srs(next_area.start_index)} - {take_srs(next_area.end_index)})"
    cv2.putText(frame_to_display, display_text, (150,290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow(WIN_X, frame_to_display)

    cv2.waitKey(30) # refresh

# for i in tqdm.tqdm(range(200)):
#     _, fref = cap.read()
#     cv2.imshow(WIN_X, fref)
#     k = cv2.waitKey(30)

# create_window(WIN_X)
# shoow_frame(WIN_X)
# cv2.waitKey(3000) # move window
# cv2.destroyWindow(WIN_X)

# smooth_sum(i)
# frames_to_time(i)

label = np.zeros(shape=(timmy_vid_letsgo.shape[0],1), dtype='bool')
for x in label_frames:
    label[x[0]:x[1], ...] = True

dat = pd.DataFrame([label[:, 0], (label[:,0]!=np.roll(label[:,0], 1)).cumsum()]).T
dat.columns=['label', 'group']
summarised_labels = dat.groupby('group').agg(
    label = ('label', 'first'), 
    start_index= ('label', lambda x: x.index.min()),
    end_index = ('label', lambda x: x.index.max())
).reset_index()

import tensorflow as tf
# inputs = tf.random.normal([32, 10, 8])


frame_inputs = tf.keras.Input(shape=(300, 360, 640, 3))

# m1 = tf.keras.layers.Bidirectional(tf.keras.layers.ConvLSTM2D(3, 3, strides=4, return_sequences=True, data_format='channels_last'))(frame_inputs)
# m2 = tf.keras.layers.Bidirectional(tf.keras.layers.ConvLSTM2D(3, 3, strides=4, return_sequences=True, data_format='channels_last'))(m1)
# m3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')(m2)
# m4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(m3)
# m5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512,))(m4)
# m6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32,))(m5)
# m7 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,))(m6)
# m8 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('softmax'))(m7)


# m1.shape
# m2.shape
# m3.shape
# m4.shape
# m5.shape
# m6.shape
# m7.shape
# m8.shape

# model = tf.keras.Model(inputs = frame_inputs, outputs = m8, name='first_model')
# model.summary()

m1 = tf.keras.layers.ConvLSTM2D(3,3,strides=4, return_sequences=True, data_format='channels_last')(frame_inputs)
m2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(m1)
m3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1,))(m2)
m4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('softmax'))(m3)

model = tf.keras.Model(inputs = frame_inputs, outputs = m4, name='second_model')
model.summary()

def get_frame_batch(lab_start):
    return timmy_vid_letsgo[None, lab_start:lab_start+300, ...].copy()

def get_seq_labs(lab_start):
    return label[None, lab_start:lab_start+300,:].copy()

train_dat = np.concatenate([get_frame_batch(250 + x*5) for x in range(20)])
train_labs = np.concatenate([get_seq_labs(250 + x*5) for x in range(20)])
train_dat.shape
train_labs.shape

# report_tensor_allocations_upon_oom
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(train_dat, train_labs, epochs=1, batch_size=2)
# ?model.fit

# model.predict(train_dat)
# model.predict(train_dat[:1])
np.unique(model.predict(train_dat[:1]), return_counts=True)


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# y_test



np.unique(seq_labels[0,:,0], return_counts=True)
np.unique(model.predict(single_batch), return_counts=True)

seq_labels.shape

seq_out, final_state = m1(frame_inputs)
inputs.shape
seq_out.shape
timmy_vid_letsgo
timmy_vid_letsgo


(2*60+55)*2

cv2.destroyWindow(WIN_X)

print()
k

out_vid = np.array(full_vid)

save_gif(out_vid[-100:], 30)

cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)

cap.get(cv2.CAP_PROP_POS_FRAMES)
dir(cap)

dir(cv2)
dir(cv2)

