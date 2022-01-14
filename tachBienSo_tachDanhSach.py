import cv2
import numpy as np
import os

path_folder = "./dataTest/"
dem = 0
file_list = os.listdir(path_folder)
for img_item in file_list:
    print(img_item)
    img = cv2.imread(os.path.join(path_folder, img_item))
    # img = cv2.imread("./dataTest/0329_03016_b.jpg")
    # img = cv2.resize(img, (620, 480))
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
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        if len(approx) == 4 and 1.1 <= aspectRatio <= 1.5:
            screenCnt = approx
            break

    if screenCnt is not None:
        anh_kytu = img[y:(y + h), x:(x + w)]
        anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_RGB2GRAY)
        anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("alo", anh_kytu)
        # if not os.path.exists('dataCharacters/'):
        #     os.mkdir('dataCharacters/')
        # cv2.imwrite('dataCharacters/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
        # dem = dem + 1

        # cv2.imshow("alo", anh_kytu)
        if not os.path.exists('dataLinhTinh/'):
            os.mkdir('dataLinhTinh/')
        anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
        cv2.imwrite('dataLinhTinh/' + img_item, anh_kytu_bw)
        dem = dem + 1
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("bienso", img)
        # cv2.waitKey(0)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
