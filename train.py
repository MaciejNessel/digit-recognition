import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
from model import create_model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


img_arr, c_name = utils.import_images('data')

X_train, X_test, y_train, y_test = train_test_split(img_arr, c_name, train_size=0.65, test_size=0.35)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.65, test_size=0.35)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_validation = to_categorical(y_validation, 10)

my_model = create_model()

history = my_model.fit_generator(dataGen.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train) // 32,
                                 epochs=10, validation_data=(X_validation, y_validation), shuffle=1)


score = my_model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

my_model.save('my_model.h5')
