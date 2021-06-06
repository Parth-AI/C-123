from pickle import TRUE
import cv2
import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context'), None):
     ssl._create_default_https_context = ssl._create_unverified_context

X,Y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(Y).value_counts())
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 9, train_size = 7500, test_size = 25)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class= 'multinomial').fit(X_train_scale, Y_train)

y_predict = clf.predict(X_test_scale)
accuracy = accuracy_score(Y_test, y_predict)
print(accuracy)

cap = cv2.VideoCapture(0)

while True:
     try:
          ret, frame = cap.read()
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          height, width = gray.shape
          upper_left = (int(width/2-56), int(height/2-56))         
          bottom_right = (int(width/2+56), int(height/2+56))
          cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)        
          roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
          im_PIL = Image.fromarray(roi)
          img_bw = im_PIL.convert('L')
          img = img_bw.resize((28, 28), Image.ANTIALIAS)
          img_inverted = PIL.ImageOps.invert(img)
          pixel_filter = 20
          min_pixel = np.percentile(img_inverted, pixel_filter)
          img_inverted_scale = np.clip(img_inverted-min_pixel, 0, 255)
          max_pixel = np.max(img_inverted)
          img_inverted_scale = np.asarray(img_inverted_scale)/max_pixel
          test_sample = np.array(img_inverted_scale).reshape(1, 784)
          test_pred = clf.predict(test_sample)
          print(test_pred)
          cv2.imshow('frame', gray)
          if cv2.waitKey(1) & 0xff == ord('q'):
               break
     except Exception as e:
          pass

cap.release()
cv2.destroyAllWindows()
