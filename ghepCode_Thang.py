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


class License_plate_code:
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


def detectPlate():
    img = cv2.imread("./reservedData/381.jpg")
    # img = cv2.resize(img, dsize=(472, 303))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)[1]
    canny_image = cv2.Canny(thresh_image, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    # cv2.imwrite('dataLinhTinh/' + "1.jpg", dilated_image)
    contours = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        if len(approx) == 4 and 1.1 <= aspectRatio <= 1.5:
            screenCnt = approx
            break

    if screenCnt is None:
        print("Can't detect!")
        return

    anh_kytu = img_gray[y:(y + h), x:(x + w)]
    # Thực hiện chuyển đổi ảnh đen trắng với ngưỡng 210
    anh_kytu_bw = cv2.threshold(anh_kytu, 170, 255, 3)[1]
    anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
    segmentChar(anh_kytu_bw)


def segmentChar(img):
    position = []
    list_img = []
    dem = 0
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
            anh_kytu = img[y:(y + h), x:(x + w)]
            anh_kytu = cv2.resize(anh_kytu, dsize=(112, 112))
            list_img.append(anh_kytu)
            position.append([x, y, dem])
            dem += 1
    sort_position(position)
    predictChar(list_img)


def predictChar(list_img):
    global strPlate
    global first_line
    global second_line
    global saved_model

    list_str_char = []
    class_plate = License_plate_code()
    for imgChar in list_img:
        img_cvt_rgb = cv2.cvtColor(imgChar, cv2.COLOR_GRAY2RGB)
        img_cvt_rgb = np.array(img_cvt_rgb)
        result = saved_model.predict(img_cvt_rgb.reshape(1, 112, 112, 3))
        final = np.argmax(result)
        final = class_plate.classNames[final]
        list_str_char.append(final)
    list_str_char = np.array(list_str_char)

    list_char = np.concatenate([list_str_char[first_line[:, 2]], list_str_char[second_line[:, 2]]])
    strPlate = list_char[0] + list_char[1] + '-' + list_char[2] + list_char[3] + ' '
    if np.count_nonzero(list_char) == 9:
        strPlate += list_char[4] + list_char[5] + list_char[6] + '.' + list_char[7] + list_char[8]
    elif np.count_nonzero(list_char) == 8:
        strPlate += list_char[4] + list_char[5] + list_char[6] + list_char[7]
    else:
        strPlate = "Can't detect"
    print('Bien so xe: ' + strPlate)


def sort_position(position):
    global first_line
    global second_line
    position = np.array(position)
    position = position[np.argsort(position[:, 1])]
    # print(position)

    first_line = position[0:4]
    second_line = position[4:]
    first_line = first_line[np.argsort(first_line[:, 0])]
    second_line = second_line[np.argsort(second_line[:, 0])]


if __name__ == '__main__':
    # Tách ký tự
    saved_model = tf.keras.models.load_model("./reservedData/BienSo_112_112.h5")
    # position = []
    first_line = []
    second_line = []
    timeStart = datetime.now()
    strPlate = ''
    detectPlate()
    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)
