import os
import pickle
import cv2
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

print("Bắt đầu định dạng lại ảnh")

df = pd.DataFrame({'path': [], 'label': []})
classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
              "19", "20", "21", "22", "23", "24", "25", "26", "27",
              "28", "29", "30"]
for name in classnames:
    img_path = "dataChar/" + str(name) + "/"
    filelist = os.listdir(img_path)
    for file in filelist:
        if file != 'Thumbs.db':
            img = os.path.join(img_path, file)
            new_row = {'path': img, 'label': name}
            df = df.append(new_row, True)

df.to_csv(r'dataBeforeTraining/pathImage56.csv', index=False)


list_data = []
data = pd.read_csv('dataBeforeTraining/pathImage56.csv')
X = data['path'].tolist()
Y = data['label'].tolist()

for i in X:
    print(i)
    img = cv2.imread(i)
    image = cv2.resize(img, (56, 56))
    list_data.append(image)

list_data = np.array(list_data)

# bien doi nhan
le = preprocessing.LabelEncoder()
le.fit(Y)
label = le.transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
label = label.reshape(len(label), 1)
onehot_encoded = np.array(onehot_encoder.fit_transform(label))

print(onehot_encoded)
# luu lai

pickle.dump(le, open('dataBeforeTraining/decode_label_56.pkl', 'wb'))
pickle.dump(onehot_encoded, open("dataBeforeTraining/one_hot_label_56.pkl", "wb"))
pickle.dump(list_data, open('dataBeforeTraining/data_56.pkl', 'wb'))

print('done')
