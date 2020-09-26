from keras.models import load_model, save_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import itertools

classifier = load_model("lessArchsplitlettnum.hdf5")
num_classes = 36
PATH = os.getcwd()
data_path = './num_letters/scaled_test'
data_path2 = './num_letters/scaled_train'
data_dir_list = os.listdir(data_path) #direktoriji unutra
data_dir_list2 = os.listdir(data_path2)

img_rows=28
img_cols=28
num_channel=1 #rgb

num_epoch=1 #treniranje

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

y_pred = classifier.predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Get current size
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print("Current size:", fig_size)

# Set figure width to 12 and height to 9
fig_size[0] = 30
fig_size[1] = 30
plt.rcParams["figure.figsize"] = fig_size
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()