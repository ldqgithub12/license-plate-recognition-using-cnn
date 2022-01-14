import cv2
import numpy as np
import os
import tensorflow as tf

img = cv2.imread("dataTest/6.jpg")
img = cv2.resize(img, dsize=(472, 303))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
equal_histogram = cv2.equalizeHist(noise_removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
sub_morp_image = cv2.subtract(equal_histogram, morph_image)
ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image, 250, 255)
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
dem = 0
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspectRatio = w / float(h)
    if len(approx) == 4 and 1.1 <= aspectRatio <= 1.5:
        screenCnt = approx
        anh_cut = img[y:(y + h), x:(x + w)]
        anh_kytu_gray = cv2.cvtColor(anh_cut, cv2.COLOR_RGB2GRAY)
        anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
        anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
        cv2.imwrite('Biensocut/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
        break
dem = dem + 1

# Tách ký tự
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
path_folder = "Biensocut/"
dem = 0
images = []
labels = []
toado = []
ketqua = []
file_list = os.listdir(path_folder)
for img_item in file_list:
    img = cv2.imread(os.path.join(path_folder, img_item))
    labels.append(str(img_item))
    images.append(img)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # dem = 1
    label = str(img_item)
    label = label[:(len(label) - 4)]
    saved_model = tf.keras.models.load_model("Data/BienSo7.h5")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
            anh_kytu = img[y:(y + h), x:(x + w)]
            anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_RGB2GRAY)
            anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
            toado.append([x, y + h])
            anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(112, 112))
            result = saved_model.predict(np.array(anh_kytu_bw))
            final = np.argmax(result)
            ketqua.append(result)
            # if not os.path.exists('kytucut/'):
            #     os.mkdir('kytucut/')
            # anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(112, 112))
            # cv2.imwrite('kytucut/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
            # dem = dem + 1
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        # cv2.imshow("demo", char_number)

# nhan diện ký tự
cv2.waitKey(0)
print("Done")
