import cv2
src="haarcascade_frontalface_default.xml"
face=cv2.CascadeClassifier(src)
cam=cv2.VideoCapture(0)

while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("img",img)
    key=cv2.waitKey(1)
    if key==ord("s"):
        break
    
cam.release()
cv2.destroyAllWindows()
