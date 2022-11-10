import numpy as np
import idx2numpy
#import matplotlib.pyplot as plt
import gzip as gz
from PIL import Image
import tensorflow as tf
import tensorflow.python.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model

from API import *

tf.random.set_seed(314)

read_file()


train_imagefile = 'train-images-idx3-ubyte'
train_imagearray = idx2numpy.convert_from_file(train_imagefile)

test_imagefile = 't10k-images-idx3-ubyte'
test_imagearray = idx2numpy.convert_from_file(test_imagefile)

train_labels = get_labels('train')
test_labels = get_labels('test')

#train_imagearray shape = (60000, 28, 28)
#labels unique values =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#test_imagearray shape = (10000,28,28)


inputs = (28,28,1)
batch_size = 128
epochs = 100
learning_rate = 0.001

train_encoding = np.eye(len(set(list(train_labels))))[train_labels]
test_encoding = np.eye(len(set(list(test_labels))))[test_labels]

num_class = len(set(train_labels.tolist()))

##원본 데이터
X_TRAIN,X_TEST,Y_TRAIN,Y_TEST = train_imagearray,test_imagearray, train_encoding,test_encoding

##학습용 데이터
X_train, X_val,y_train,  y_val = train_test_split(X_TRAIN, Y_TRAIN,
                                                  test_size=0.2, shuffle=True, stratify=Y_TRAIN, random_state=34)


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_TEST = X_TEST.astype('float32')

X_train /= 255
X_val /= 255
X_TEST /= 255

X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
X_TEST = X_TEST.reshape(-1,28,28,1)




steps_per_epoch = X_train.shape[0] // batch_size
validation_steps = int(np.ceil(X_val.shape[0]/batch_size))

##TRAIN DATA SET
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(500).\
    batch(batch_size, drop_remainder=True).repeat()
#TEST DATA SET
test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)



model = create_model_CNN(input=inputs,category=num_class)

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
hist = model.fit(train_dataset,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=test_dataset,
                 validation_steps=validation_steps)


score = model.evaluate(X_TEST, test_encoding, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


##모델 저장
model.save('cnn_model_weight.h5')














































