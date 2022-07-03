import tensorflow as tf
import pretty_midi
import numpy as np
import librosa
import librosa.display
from os.path import isfile
import multiprocessing as mp
from queue import Empty
import glob
from math import floor

ENABLE_THREADING = True
#DATA_ROOT = '/home/jupyter/e-gmd-v1.0.0/'
DATA_ROOT = '/home/jupyter/test/'
OUTPUT_ROOT = '/home/jupyter/'
TRAIN_SPLIT = 0.7 # 70% training data, 30% validation

# Keep in sync with cell 1 of main.ipynb
spec_feature_count = 128
drum_notes = [35, 38, 42, 46, 41, 43, 45, 47, 48, 50, 49, 51]
output_classes = len(drum_notes)

segment_length_secs = 5
segment_length = 215 #librosa.time_to_frames(segment_length_secs)

# TODO: data augmentation using audiomentations: https://github.com/iver56/audiomentations
# TODO: pad data with zero_padding / silence

# Maps notes into 12 drum classes (kick, snare, closed hihat, open hihat, 6x toms, crash cymbal, ride cymbal)
gm_drum_map = {
    # kicks
    35: 35,
    36: 35,
    # snares
    38: 38,
    40: 38,
    # hihats
    42: 42,
    44: 42,
    22: 42,
    # open hihat
    46: 46,
    26: 46,
    # toms
    41: 41,
    43: 43,
    45: 45,
    47: 47,
    48: 48,
    50: 50,
    # crash / splash cymbals
    49: 49,
    55: 49,
    52: 49,
    57: 49,
    # ride cymbals
    51: 51,
    59: 51,
}

class_to_idx = {k: v for v, k in enumerate(drum_notes)}
thread_count = mp.cpu_count()
print(f'CPU count: {thread_count}')
high_pass_hz = 15000
high_pass_ratio = 40

def samples_to_np(a):
    return np.array(a.get_array_of_samples()).astype(np.float32) / 32767.0

def process_file(data_file, callback):
    label_file = data_file[:-4] + '.midi'
    if not isfile(data_file) or not isfile(label_file):
        return
    audio, sr = librosa.load(data_file)
    if np.isnan(audio).any():
        raise Exception('found nan in transformed audio clip')
    stft_features = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=spec_feature_count, n_fft=2048, hop_length=512))
    if np.isnan(stft_features).any():
        raise Exception('found nan in stft')

    pm = pretty_midi.PrettyMIDI(label_file)
    instrument = pm.instruments[0]
    pitches = []
    onsets = []

    for i, note in enumerate(instrument.notes):
        mapped_note = gm_drum_map.get(note.pitch, 0)
        if mapped_note == 0:
            continue
        pitches.append(mapped_note)
        onsets.append(note.start)

    onset_frames = librosa.time_to_frames(onsets, sr=sr)
    labels = np.zeros((output_classes, stft_features.shape[1]), dtype='int64')

    for i in range(len(onsets)):
        frame = onset_frames[i]
        if frame >= labels.shape[1]:
            continue
        class_idx = class_to_idx.get(pitches[i])
        labels[class_idx][frame] = 1
        # Soft target vectors
        if frame > 0:
            labels[class_idx][frame - 1] = 0.5
        if frame < labels.shape[1] - 1:
            labels[class_idx][frame + 1] = 0.5

    segment_count = stft_features.shape[1] // segment_length
    for i in range(segment_count):
        stft_slice = stft_features[:, i * segment_length:(i + 1) * segment_length]

        labels_slice = labels[:, i * segment_length:(i + 1) * segment_length]
        if stft_slice.shape[1] != labels_slice.shape[1]:
            raise Exception('mismatched total frame count between stft_features and labels')
        callback(stft_slice, labels_slice)

def _write_tfrecords_thread(writer, file):
    def write_record(stft_slice, labels_slice):
        feature = {
            'stft': tf.train.Feature(float_list=tf.train.FloatList(value=stft_slice.flatten())),
            'labels': tf.train.Feature(float_list=tf.train.FloatList(value=labels_slice.flatten())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    process_file(file, write_record)

def write_tfrecords(data_files, tfrecord_path_prefix):
    if not ENABLE_THREADING:
        _write_tfrecords_thread(tfrecord_path_prefix + '.tfrecords', data_files)
    else:
        q = mp.Queue()
        for file in data_files:
            q.put(file)

        def worker(id):
            tfrecord_path = tfrecord_path_prefix + str(id) + '.tfrecords'
            with tf.io.TFRecordWriter(tfrecord_path) as writer:
                while True:
                    try:
                        file = q.get(True)
                        _write_tfrecords_thread(writer, file)
                    except Empty:
                        writer.flush()
                        writer.close()
                        return

        processes = []
        for i in range(thread_count):
            p = mp.Process(target=worker, args=[i])
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print('Done')

all_files = glob.glob(DATA_ROOT + '*.wav', recursive=True)
split = floor(len(all_files) * TRAIN_SPLIT)

train_files = all_files[:split]
validate_files = all_files[split:]

print(f'Found {len(train_files)} files for training and {len(validate_files)} for validation')
print('Processing training files now')
write_tfrecords(train_files, OUTPUT_ROOT + 'train')
print('Processing validation files now')
write_tfrecords(validate_files, OUTPUT_ROOT + 'validate')
