
import pyscreenshot as ImageGrab
import time

# Evaluation the model  
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score ,auc ,f1_score ,roc_curve
from sklearn.metrics import precision_score, recall_score


 
#built a model           
        
  #load the dataset
import pandas as pd
from sklearn.utils import shuffle
data  =pd.read_csv('dataset.csv')
data=shuffle(data)
data    

#separation of dependent and independent variable
X = data.drop(["label"],axis=1)
Y= data["label"]
 
#preview of one image using matplotlib

import matplotlib.pyplot as plt
import cv2
idx = 314
img = X.loc[idx].values.reshape(28,28)
print(Y[idx])
plt.imshow(img)
 
#  #Train-Test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)
 
# #Fit the model using svc and also to save the model using joblib
import joblib
from sklearn.svm import SVC
classifier=SVC(kernel="linear", random_state=6)
classifier.fit(x_train,y_train)
joblib.dump(classifier, "model/digit_recognizer")
 
#calculate accuracy
from sklearn import metrics
test_pred=classifier.predict(x_test)
print("Accuracy= ",metrics.accuracy_score(test_pred,y_test)) 
print("\n")

# confusion matrix and classification report:
    
cm = confusion_matrix(y_test, test_pred)
print('confusion matrix:')
print(cm)
print("\n")
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
rf=classification_report(y_test, test_pred)
print('classification report is:')
print(rf)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

from sklearn import scikitplot as skplt
skplt.metrics.plot_roc_curve(test_pred,y_test)


   
# #####################################################################################

# #prediction of image drawn in paint
 
# import joblib
# import cv2
# import numpy as np #pip install numpy
# import time
# import pyscreenshot as ImageGrab
 
# model=joblib.load("model/digit_recognizer")
# images_folder="img/"
 
# while True:
#     img=ImageGrab.grab(bbox=(60,170,400,500))
   
#     img.save(images_folder+"img.png")
#     im = cv2.imread(images_folder+"img.png")
#     im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     im_gray  =cv2.GaussianBlur(im_gray, (15,15), 0)
   
#     #Threshold the image
#     ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
#     roi = cv2.resize(im_th, (28,28), interpolation  =cv2.INTER_AREA)
   
#     rows,cols=roi.shape
   
#     X = []
   
#     ## Add pixel one by one into data array
#     for i in range(rows):
#         for j in range(cols):
#             k = roi[i,j]
#             if k>100:
#                 k=1
#             else:
#                 k=0
#             X.append(k)
           
#     predictions  =model.predict([X])
#     print("Prediction:",predictions[0])
#     cv2.putText(im, "Prediction is: "+str(predictions[0]), (20,20), 0, 0.8,(0,255,0),2,cv2.LINE_AA)
   
#     cv2.startWindowThread()
#     cv2.namedWindow("Result")
#     cv2.imshow("Result",im)
#     cv2.waitKey(10000)
#     if cv2.waitKey(1)==13: #27 is the ascii value of esc, 13 is the ascii value of enter
#         break
# cv2.destroyAllWindows()
 


  
 
