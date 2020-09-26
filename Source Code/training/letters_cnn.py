import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import save_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K

K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD,RMSprop,adam

num_classes = 36 #A-Z

def get_model():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=input_shape, padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

PATH = os.getcwd()
data_path2 = './num_letters/scaled_test'
data_path = './num_letters/scaled_train'
data_dir_list = os.listdir(data_path) #direktoriji unutra
data_dir_list2 = os.listdir(data_path2)

img_rows=28
img_cols=28
num_channel=1 #rgb

num_epoch=25 #treniranje

img_data_list=[]
label_numbers = []
label_numbers.append(0)
labels = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    i = 0
    for img in img_list:
        input_img=cv2.imread(data_path + '\\'+ dataset + '\\'+ img )
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_data_list.append(input_img)
        i = i + 1
        labels.append(int(dataset))
    print(i)
for dataset in data_dir_list2:
    img_list=os.listdir(data_path2+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    i = 0
    for img in img_list:
        input_img=cv2.imread(data_path2 + '\\'+ dataset + '\\'+ img )
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_data_list.append(input_img)
        i = i + 1
        labels.append(int(dataset))
    print(i)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

img_data= np.expand_dims(img_data, axis=4)
print(img_data.shape)

num_of_samples = img_data.shape[0]

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

input_shape = img_data[0].shape

model = get_model()

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

hist = model.fit(X_train, y_train, batch_size=128, epochs=num_epoch, verbose=1, validation_split=0.1)

model.save('letters.hdf5')

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val']) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.show()




#%%

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(y_test[1:5])



#%%


# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print(classification_report(np.argmax(y_test, axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))