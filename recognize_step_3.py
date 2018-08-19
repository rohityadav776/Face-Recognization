
import cv2
import numpy as np
#faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
faceDetect=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml');
#faceDetect=cv2.CascadeClassifier('cars.xml');

print("hello")
count=0
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor=(0,255,0)

cam = cv2.VideoCapture('ironman.mp4')

#cam=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingdata.yml")
id=0

#font=cv2.InitFont(cv2.CV_font_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
    ret,img=cam.read();
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("hello")
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    #faces=faceDetect.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id='Robert Downey'
        else:
            id="sorry false detection"
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
        cv2.putText(img,str(id),(x,y+h),fontface,fontscale,fontcolor)
        
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()  


print("count : ",count)
