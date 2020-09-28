import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
size = 4
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        mirror = cv2.flip(frame, 1)
        mini = cv2.resize(mirror, (mirror.shape[1] // size, mirror.shape[0] // size))
        gray = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 5)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(mirror, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', mirror)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
