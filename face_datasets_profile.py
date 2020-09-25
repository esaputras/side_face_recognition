import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
 
vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_profileface.xml')
assure_path_exists("dataset/")

while(True):
    name = input("Siapa? ")
    count = 0

    if (name == "q"):
        break
    ids = input("id? ")
    side = input("sisi? ")
    
    while(True):        
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        
        if(side == 'kiri'):
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
        else :
            right_gray = cv2.flip(gray, 1)
            faces = face_detector.detectMultiScale(right_gray, 1.3, 5)
        
        #faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            #cv2.flip(image_frame,0)
            right_image_frame = cv2.flip(image_frame, 1)
            if(side != 'kiri'):
                cv2.rectangle(right_image_frame,(x,y),(x+w,y+h), (255,0,0), 2)
            else:
                cv2.rectangle(image_frame,(x,y),(x+w,y+h), (255,0,0), 2)
            #cv2.rectangle(image_frame, (x+w,y+h), (x,y), (255,0,0), 2)
            count += 1
            #cv2.imwrite("dataset profile/" + name + '-' + side + '.' + ids + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            if(side == 'kiri'):
                cv2.imwrite("dataset/" + name + '-' + side + '.' + ids + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame', image_frame)
            else:
                #flipgray = cv2.flip(right_gray,1)
                cv2.imwrite("dataset/" + name + '-' + side + '.' + ids + '.' + str(count) + ".jpg", cv2.flip(right_gray[y:y+h,x:x+w],1))
                cv2.imshow('frame', right_image_frame)
            cv2.waitKey(100)
            print (count)
        if(side == 'kiri'):
            if(count > 34 ):
                break
        else :
            if(count > 34):
                break
        
        
vid_cam.release()
cv2.destroyAllWindows()
