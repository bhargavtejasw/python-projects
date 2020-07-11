import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils as npu
from keras.backend import clear_session

# Load Model
(train_X , train_y), (test_X , test_y) = mnist.load_data("mymnist.data")
# Reshape data and change type
test_X = test_X.reshape(-1 , 28, 28, 1)
train_X = train_X.reshape(-1 ,  28, 28, 1)
test_X = test_X.astype("float32")
train_X = train_X.astype("float32")
# One hot encoding
test_y = npu.to_categorical(test_y)
train_y = npu.to_categorical(train_y)

accuracy= open("accuracy.txt","r")
accuracy = float(accuracy.read())
accuracy = accuracy *100
#Initials
neurons = 10
epochs = 1
test = 1
flag = 0
kernel = 8
batch_size = 128
#filter = 3


while int(accuracy)<90:
    if flag == 1 :
        model = keras.backend.clear_session()
        neurons = neurons+10
        epochs = epochs+1
        test = test + 1
        kernel = kernel * 2
        test = test + 1
    print("* * * TRIAL : ",test ,"-----------------")
    model=Sequential()
    model.add(Conv2D(kernel, (3,3), input_shape = (28, 28, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Flatten())
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile( optimizer= "Adam" , loss='categorical_crossentropy',  metrics=['accuracy'] )
    train_X.shape
    model_predict= model.fit(train_X, train_y,batch_size=batch_size,verbose=1,epochs=epochs,validation_data=(test_X, test_y),shuffle=True)
    scores = model.evaluate(test_X, test_y, verbose=False)
    print('Test loss :', scores[0]*100)
    print('-------Accuracy of the model :', scores[1]*100)
    accuracy = scores[1]*100
    print("_______________________________________________________")
    print()
    print()
    flag = 1

print("Total numbers of epochs :" , epochs)
print("Total number of filters :", kernel)
print("Total number of neurons :", neurons)
print("Final Accuracy : ", accuracy)


import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()

s.login("anshshriv22296@gmail.com", "********")


    # message
message_success = "Achieved your desired accuracy . Congrats "


    # sending the mail
s.sendmail("anshshriv22296@gmail.com", "rajansh87@gmail.com", message_success)


    # terminating the session
s.quit()