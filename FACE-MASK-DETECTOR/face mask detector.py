import cv2
import numpy as np
from keras.models import load_model
model=load_model("./model2-004.model")

labels_dict = {0: 'without mask', 1: 'mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

size = 4
webcam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    (rval, im) = webcam.read()
    mirror=cv2.flip(im, 1, 1)
    mini = cv2.resize(mirror, (mirror.shape[1] // size, mirror.shape[0] // size))
    faces = classifier.detectMultiScale(mini)

    # Drawing rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = mirror[y:y+h, x:x+w]
        resized=cv2.resize(face_img, (150, 150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        cv2.rectangle(mirror, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(mirror, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(mirror, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255,255), 2)

    cv2.imshow('WEBCAM', mirror)
    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
