import os
import cv2
import numpy as np
from model import create_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def img_preprocess(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def import_images(path):
    images = []     
    class_name = []

    for digit in range(10):
        digit_dict = os.listdir(path+"/"+str(digit))
        for digit_img in digit_dict:
            img = cv2.imread(path+"/"+str(digit)+"/"+digit_img)
            img = img_preprocess(img)

            images.append(img)
            class_name.append(digit)

    result_images = np.array(images)
    result_class_name = np.array(class_name)
    return result_images, result_class_name


img_arr, c_name = import_images('data')

# SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(img_arr, c_name, train_size=0.65, test_size=0.35)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.65, test_size=0.35)

# RESHAPE IMAGES
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_validation = to_categorical(y_validation, 10)

my_model = create_model()

# STARTING THE TRAINING PROCESS
history = my_model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train) // 32,
                                 epochs=10, validation_data=(X_validation, y_validation), shuffle=1)


# EVALUATE USING TEST IMAGES
score = my_model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

# SAVE THE TRAINED MODEL
my_model.save('my_model.h5')
