# import libraries:

import os
import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix ,roc_curve , roc_auc_score
from sklearn.metrics import classification_report

# load dataset

dataset = pd.read_csv('dataset.csv')
x= dataset.iloc[:,1:].values
y=dataset.iloc[:, 0]

# split the dataset to train and test

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# implementing the ann

model =Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size = 10,epochs=5)

model.save('handwritten.model')

model =load_model('handwritten.model')

# loss curve

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# roc curve:

lr_probs = model.predict(x_test)

# calculate scores

lr_auc = roc_auc_score(y_test, lr_probs ,multi_class='ovr')

ir_fpr={}
ir_tpr={}
thresh={}
# calculate roc curves
for i in range(10):
    lr_fpr, lr_tpr,thresh[i]= roc_curve(y_test, lr_probs[:,i],pos_label=i)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Number({i})'.format(i=i))
    
print(' ROC AUC=%.3f' % (lr_auc))

# plot the roc curve for the model

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC:')
plt.legend()
plt.show()


# predict the test

test_pred = pd.DataFrame(model.predict(x_test, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

print("Accuracy:",accuracy_score(y_test, test_pred))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# confusion matrix and classification report:
    
cm = confusion_matrix(y_test, test_pred)
print('confusion matrix:')
print(cm)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
cr=classification_report(y_test, test_pred)
print('classification report is:')
print(cr)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# read images from device and predict it

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img= cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    except:
        print("Error")
    finally:
        image_number +=1



