


from PIL import Image




import os




import numpy as np




import cv2
import cv2.face




recognizer = cv2.face.LBPHFaceRecognizer_create()




path='dataset'
def getImageWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),faces   

Ids,faces=getImageWithID(path)
recognizer.train(faces,Ids)
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()

