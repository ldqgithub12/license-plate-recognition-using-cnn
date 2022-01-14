# import os
#
# # oldpath = './user/user.txt'
# # newpath = './user/user_infor.txt'
# # os.rename(oldpath, newpath)
#
# path_folder = "./dataTest/"
# dem = 0
# file_list = os.listdir(path_folder)
# for img_item in file_list:
#     pathOld = path_folder + str(img_item)
#     pathNew = path_folder + str(dem) + ".jpg"
#     dem = dem + 1
#     print("old = " + pathOld)
#     print("new = " + pathNew)
#     os.rename(pathOld, pathNew)

from datetime import datetime
import time


now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
print(type(now))
# now = datetime.strftime(now, '%d/%m/%Y %H:%M:%S')
time.sleep(5)
old = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
# old = datetime.strftime(old, '%d/%m/%Y %H:%M:%S')
# a = old - now
# a = datetime.fromordinal(a)
# if a > 5:
#     print('alo')
# else:
#     print('abc')
# print(a)
