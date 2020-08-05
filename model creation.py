# creating model 
import numpy as np 
import cv2 
import os 
import time 

file_path =  r'C:\Users\abdul\Desktop\opencv project\Facial Recognition system\face data'


def prepare_training_data(file_path):
    img_names = [file for file in os.listdir(file_path) 
                 if os.path.isfile(os.path.join(file_path, file))]
    training_data = []
    labels = []
    for i, img in enumerate(img_names):
        img_path = os.path.join(file_path, img)
        img = cv2.imread(img_path, 0)
        if img is not None:
            training_data.append(img)
            labels.append(i)
    return np.array(training_data), np.array(labels)


def train_my_model(X_train, y_train, model = cv2.face.LBPHFaceRecognizer_create()):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, y_train)
    return model 
 
    
training_data, labels = prepare_training_data(file_path)
model = train_my_model(training_data, labels)


def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_xml = r'C:\Users\abdul\Desktop\Haar Cascade github\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_xml)
    faces = face_cascade.detectMultiScale(gray)
    if faces is not None:
        for(x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 219), 2)
        return face 
    else:
        return None 

def main():
    cap = cv2.VideoCapture(0)
    
    
    while cap.isOpened():
        
        _, frame = cap.read()
        
        try:
            face = extract_face(frame)
            result = model.predict(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            print(result)
            confidence = result[1]
            if confidence < 100:
                cv2.putText(frame, 'unlocked', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 255, 0), 2)
                cv2.imshow('frame', frame)
            else:
                cv2.putText(frame, 'Locked', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 0, 255),)
                cv2.imshow('frame', frame)
                
        except Exception as e:
            cv2.putText(frame, 'Face not detected', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1 , (100, 255, 20), 2)
            cv2.imshow('frame', frame)
            
        if cv2.waitKey(1) == 27:
            break 
        
    cap.release()
    cv2.destroyAllWindows()
    
 
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    