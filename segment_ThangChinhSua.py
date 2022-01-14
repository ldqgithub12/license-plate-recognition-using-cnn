import cv2
# import matplotlib as mtl
# import self as self
from skimage import measure
import os
import numpy as np
# from skimage.filters import threshold_local
# import imutils

images = []
labels = []
# path_folder = "datasetMotorcycle/"
path_folder = "dataLinhTinh/"
dem = 0
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
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
            anh_kytu = img[y:(y+h), x:(x+w)]
            anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_RGB2GRAY)
            anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow("alo", anh_kytu)
            # if not os.path.exists('dataCharacters/'):
            #     os.mkdir('dataCharacters/')
            # cv2.imwrite('dataCharacters/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
            # dem = dem + 1

            cv2.imshow("alo", anh_kytu)
            if not os.path.exists('dataLinhTinh/'):
                os.mkdir('dataLinhTinh/')
            anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(112, 112))
            cv2.imwrite('dataLinhTinh/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
            dem = dem + 1
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("img", thresh1)
    cv2.waitKey(0)
        # cv2.imshow("demo", char_number)

cv2.waitKey(0)
