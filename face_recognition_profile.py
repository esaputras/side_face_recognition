import cv2
import numpy as np
import os
import time
from datetime import datetime

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")
recognizer.read('trainer/trainer profile.yml')
cascadePath = "haarcascade_profileface.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
duration = 10
start = time.time()
nextTime = start

diffTime = round(nextTime - start, 2)
i = 0
label = input("name: ")
position = input("position: ")
print(label)
print(position)
print("start time : "+str(datetime.now()))

while (True and diffTime < duration):
    _, im = cam.read()
    print((im))
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    right_im = cv2.flip(im,1)
    print((right_im))
    if (position == 'kiri'):
        faces = faceCascade.detectMultiScale(gray, 1.3,5)
    else:
        right_gray = cv2.flip(gray, 1)
        faces = faceCascade.detectMultiScale(right_gray, 1.3, 5)

    #cv2.imwrite("dataset/" + str(diffTime) + '.jpg', cv2.flip(,1))
    #cv2.imshow('frame', right_im)
    
    if i == 0:
        start = time.time()        
        i = i+1
        
    print("start time : "+str(datetime.now()))
    for(x,y,w,h) in faces:
        #right_im = cv2.flip(im,1)
        #right_gray = cv2.flip(gray, 1)
        #cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        if(position != "kiri"):
            cv2.rectangle(right_im, (0,0),(90,90), (0,255,0), 2)#, (x-20,y-20), (-x+w+20,y+h+20), (0,255,0), 2)
        else:
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)

        #Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if(position == "kiri"):
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        else:
            Id, confidence = recognizer.predict(cv2.flip(right_gray[y:y+h,x:x+w],1))

        if(Id==1):
            str_id = "Bunga"
        elif(Id==2):
            str_id="Dhifa"
        elif(Id==3):
            str_id="Endang"
        elif(Id==4):
            str_id="Hasanah"
        elif(Id==5):
            str_id="Bulan"
        elif(Id==6):
            str_id="Esa"
        elif(Id==7):
            str_id="Satria"
        elif(Id==8):
            str_id="Ashilla"
        elif(Id==9):
            str_id="Wildan"
        elif(Id==10):
            str_id="Raihan"
        else:
            str_id = "none"  
            #if(Id == 2):
            #   Id = "Raihan {0:.2f}%".format(round(100 - confidence, 2))
        if(position == "kiri"):
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)
        else:
            #cv2.rectangle(right_im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            #cv2.putText(right_im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)
            cv2.rectangle(right_im, (0,0), (90, 90), (0,255,0), -1)
            cv2.putText(right_im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)


        #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        #cv2.putText(im, str(str_id), (x,y-40), font, 1, (255,255,255), 3)

        
        if Id == 6:
            print("Result: " + str_id + " - TRUE")
            print(round(100 - confidence, 2))
            #print("true :" + str(datetime.now()))
        

        elif Id == 9 or Id == 1 or Id == 3 or Id == 4 or Id == 5 or Id == 2 or Id == 7 or Id == 2 or Id == 8 or Id == 10:
            print("Result: " + str_id )
            print(round(100 - confidence, 2))
            #print("false :" + str(datetime.now()))

        else:
            print("Result: unknown")
            print(round(100 - confidence, 2))
            #print("fail :" + str(datetime.now()))
            

        cv2.imshow('im',im)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
            
    print("end :" + str(datetime.now()))
    nextTime = time.time()
    diffTime = round(nextTime-start,2)
    #print(diffTime)
#print("end :" + str(datetime.now()))
        
cam.release()
cv2.destroyAllWindows()
