import cv2
import numpy as np
import math
"""Ham san
img = cv2.imread("AnhXLA/xemay3.jpg")
cv2.imshow('Original', img)
cv2.waitKey(0)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
"""

img = cv2.imread("anh2.jpg", cv2.IMREAD_GRAYSCALE)
hx = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]])
hy = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])
row = len(img)
col = len(img[0])
ahx = np.zeros_like(img)
ahy = np.zeros_like(img)
kq = np.zeros_like(img)
for i in range(1, row-1):
    for j in range(1, col-1):
        tem = np.array([[img[i-1][j-1]*hx[0][0], img[i-1][j]*hx[0][1], img[i-1][j+1]*hx[0][2]],
                       [img[i][j-1]*hx[1][0], img[i][j]*hx[1][1], img[i][j+1]*hx[1][2]],
                       [img[i+1][j-1]*hx[2][0], img[i+1][j]*hx[2][1], img[i+1][j+1]*hx[2][2]]])
        t = sum(tem[0])
        t1 = sum(tem[1])
        t2 = sum(tem[2])
        ahx[i][j] = t+t1+t2
        ahx[i][j] = math.sqrt(ahx[i][j])
        tem1 = np.array([[img[i-1][j-1]*hy[0][0], img[i-1][j]*hy[0][1], img[i-1][j+1]*hy[0][2]],
                        [img[i][j-1]*hy[1][0], img[i][j]*hy[1][1], img[i][j+1]*hy[1][2]],
                        [img[i+1][j-1]*hy[2][0], img[i+1][j]*hy[2][1], img[i+1][j+1]*hy[2][2]]])
        t3 = sum(tem1[0])
        t4 = sum(tem1[1])
        t5 = sum(tem1[2])
        ahy[i][j] = t3+t4+t5
        ahy[i][j] = math.sqrt(ahy[i][j])
kq = ahx + ahy
for i in range(0, row):
    for j in range(0, col):
        if kq[i][j] >= 20:
            kq[i][j] = 1
        else:
            kq[i][j] = 0
cv2.imshow("Ket qua", kq)
cv2.waitKey(0)





