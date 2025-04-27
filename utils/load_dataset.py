import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def integer_to_one_hot(integer, min_val, max_val):
    vector_length = max_val - min_val + 1
    one_hot_vector = [0] * vector_length
    one_hot_vector[integer - min_val] = 1
    return one_hot_vector

def cut_slides(data, window_size, predict_num):
    data_slices = []
    for i in range(0, data.shape[0], window_size):
        for j in range(window_size - predict_num + 1):
            slice = data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))
            data_slices.append(slice)
    return data_slices

# --------- 滑动窗口裁剪函数 ----------
def sliding_window_split(sequence, window_size=30, stride=10):
    slices = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        slices.append(sequence[start:end])
    return slices

def load_dataset_from_files(config):
    data_dir = config['data_dir']
    x_dataset = []
    u_dataset = []
    min_val = 2
    max_val = 7
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file is a directory
        if data_file_path.endswith('.npy') and os.path.exists(data_file_path):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data = data_dict['signals'][:, :-6]
            if x_data.shape[0]!= 3417 or x_data.shape[1]!= 68:
                print(x_data.shape)
                # print(data_file_path)
            uu_data = data_dict['signals'][:, -6:-4]

            # ErrorType = data_dict['ErrorType']
            # et = np.array(integer_to_one_hot(ErrorType, min_val, max_val))
            # et_data = np.tile(et, (len(x_data), 1))
            # u_data = np.concatenate((uu_data, et_data), axis=1)

            u_data = uu_data
            # Sample the data
            x_data = x_data[::config['sample_step'], :]
            u_data = u_data[::config['sample_step'], :]
            
            x_dataset.append(x_data)
            u_dataset.append(u_data)

    return x_dataset, u_dataset

def build_training_dataset(config, x_dataset, u_dataset):
    fs, order = 1, 5
    cutoff = config['cutoff']
    for i in range(len(x_dataset)):
        x_dataset[i] = lowpass_filter(x_dataset[i], cutoff, fs, order)
    window_size = np.shape(x_dataset[0])[0]
    predict_num = config['predict_num']
    x_data = np.concatenate(x_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    x_data_slices = cut_slides(x_data, window_size, predict_num)
    u_data_slices = cut_slides(u_data, window_size, predict_num)
    x_data_slices = np.concatenate(x_data_slices, axis=0)
    u_data_slices = np.concatenate(u_data_slices, axis=0)
    return x_data_slices, u_data_slices

# 定义Butterworth低通滤波器
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 应用低通滤波器
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y
