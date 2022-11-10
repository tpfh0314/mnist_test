import numpy as np
import idx2numpy
import gzip as gz
from PIL import Image
import tensorflow as tf

import tensorflow.python.keras as keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten,Activation,Add, GlobalAveragePooling2D
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split



def create_model_CNN(input, category):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(category, activation='softmax'))

    return model


def get_labels(kind):
    if (kind == 'test'):
        filename = 't10k-labels-idx1-ubyte.gz'
    elif (kind == 'train'):
        filename = 'train-labels-idx1-ubyte.gz'
    with gz.open(filename) as file:
        # 첫 4바이트 날려주기
        leave_number = int.from_bytes(file.read(4), byteorder='big', signed=False)
        # 이후 라벨 데이터인 4바이트 read
        label_number = int.from_bytes(file.read(4), byteorder='big', signed=False)
        label_data = file.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def read_file():
    ## mnist 데이터 load
    files = [

        "train-images-idx3-ubyte.gz",

        "train-labels-idx1-ubyte.gz",

        "t10k-images-idx3-ubyte.gz",

        "t10k-labels-idx1-ubyte.gz"]

    for f in files:
        gz_file = f

        raw_file = f.replace(".gz", "")  # 파일명 변경

        print("gzip:", f)

        with gz.open(gz_file, "rb") as fp:  # gzip을 이용해서 gz파일 압축 해제.

            body = fp.read()

            with open(raw_file, "wb") as w:  # 위에서 압축이 해제된 파일을 해당 파일 명으로 생성.

                w.write(body)

