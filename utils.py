import cv2

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
