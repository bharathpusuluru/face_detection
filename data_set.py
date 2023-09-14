import cv2
import os

face_cascade = cv2.CascadeClassifier("C:/Users/bhara/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

person_name = input("Enter the name of the person: ")
save_path = f'./dataset/{person_name}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f"{save_path}/User_{str(count)}.jpg", gray[y:y+h, x:x+w])

    cv2.imshow('Image Capture', frame)

    # Break when 100 images have been captured
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
