import cv2
import os
import numpy as np

# Path to the dataset directory
dataset_path = './dataset'

# Prepare the data and labels
images = []
labels = []
label_dict = {}
current_id = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            
            # Create a label ID for each person
            if label not in label_dict:
                label_dict[label] = current_id
                current_id += 1
            
            id_ = label_dict[label]
            pil_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image_array = np.array(pil_image, "uint8")
            images.append(image_array)
            labels.append(id_)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))
recognizer.save('face-trainner.yml')
