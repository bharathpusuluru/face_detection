import cv2

# Initialize the face detector and face recognizer
face_cascade = cv2.CascadeClassifier("C:/Users/bhara/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.read('face-trainner.yml')


label_map = {0: "bharath"}


# Start capturing video
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
 
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        
        if conf <= 100:
            name = label_map.get(id_, "Unknown")
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop if 'b' key is pressed
    if cv2.waitKey(1) == ord('b'):
        break

video_capture.release()
cv2.destroyAllWindows()
