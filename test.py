import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import utils

model = load_model('my_model.h5')
default_camera = 0




cap = cv2.VideoCapture(default_camera)

while True:
    _, frame = cap.read()

    img = np.asarray(img)
    curr_img = utils.img_preprocess(frame)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(curr_img)
    recognized_digit = np.argmax(predictions, axis=1)

    probability = np.amax(predictions)

    result = f"{recognized_digit}, {probability}"

    print(result)

    if probability > 0.65:
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()