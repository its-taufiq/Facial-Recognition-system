# here we are going to see that how we can collect the sample using opencv for facial recognition 

import cv2 
import numpy as np 


def extract_face(img):
    face_xml = r'C:\Users\abdul\Desktop\Haar Cascade github\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_xml)
    # detecting faces 
    faces = face_cascade.detectMultiScale(img)
    if faces is():
        return None 
    else:
        # print(faces)
        for (x, y, w, h) in faces:
            cropped = img[y:y+h, x:x+w]
        return cropped 
       
           
    
def collect_sample():
    
    cap = cv2.VideoCapture(0)
    
    count = 0
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = extract_face(frame)
        try:
            face = cv2.resize(face, (400, 400))
        
            if face is not None:
                count = count+1
                path = r'C:\Users\abdul\Desktop\opencv project\Facial Recognition system\face data\user' + str(count) + '.jpg'
                cv2.imwrite(path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX,1, (0, 255, 0), 3 )
            else:
                print('face not detected...be stable')
        except:
            pass 
        
        if face is not None:
            cv2.imshow('detected face', face)
        
            if cv2.waitKey(1) == 13 or count == 50:
                break 
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    collect_sample()
            
            