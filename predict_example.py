import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

# print(tf.__version__)
classNames = {0: '0',
              1: '1',
              2: '2',
              3: '3',
              4: '4',
              5: '5',
              6: '6',
              7: '7',
              8: '8',
              9: '9',
              10: 'A',
              11: 'B',
              12: 'C',
              13: 'D',
              14: 'E',
              15: 'F',
              16: 'G',
              17: 'H',
              18: 'K',
              19: 'L',
              20: 'M',
              21: 'N',
              22: 'P',
              23: 'R',
              24: 'S',
              25: 'T',
              26: 'U',
              27: 'V',
              28: 'X',
              29: 'Y',
              30: 'Z',
              }

saved_model = tf.keras.models.load_model("./reservedData/BienSo_112_112.h5")

timeStart = datetime.now()
print(datetime.now())
bienanh = cv2.imread('./dataLinhTinh/motorcycle0.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('0 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle1.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('1 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle2.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('2 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle3.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('3 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle4.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('4 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle5.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('5 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle6.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('6 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle7.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('7 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./dataLinhTinh/motorcycle8.jpg')
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 112, 112, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('8 nhãn = ', final)
# plt.imshow(bienanh)

timeStop = datetime.now()
print(datetime.now())

print('timeRun = ', timeStop - timeStart)
