#main OpenCV library used for image processing and camera handling
#helper library that simplifies image transformations (like resizing)
import cv2
import imutils

#initializing the camera with id
cam=cv2.VideoCapture(0)

#initializing the first frame and area
initial_frame=None
#Minimum contour area to be considered a real motion (to ignore noise/small movements)
area=500

while True:
    #reading the frame from the camera
    __,frame=cam.read()
    text="No moving object detected"

    #Resizes the frame to a fixed width of 500 pixels for faster and consistent processing
    resize_img=imutils.resize(frame,width=500)

    #to simplifies motion detection
    gray=cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)

    #Smoothens the grayscale image to reduce noise and detail that may cause false detections
    blur=cv2.GaussianBlur(gray,(21,21),0)

    #if the initial frame is not set, set it
    if initial_frame is None:
        initial_frame=blur
        continue
    
    #calculating the difference between initial and current frame
    diff=cv2.absdiff(initial_frame,blur)

    #Converts the difference image into a binary image (black & white) where motion areas are white
    thresh=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    #fills small gaps in the detected motion regions, making contours clearer
    thresh=cv2.dilate(thresh,None,iterations=2)

    #Finds external contours in the thresholded image
    cont=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont=imutils.grab_contours(cont)

    for i in cont:
        if cv2.contourArea(i)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(i)
        cv2.rectangle(resize_img,(x,y),(x+w,y+h),(0,0,255),2)
        text="Moving object detected"
        print(text)
    cv2.putText(resize_img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("camera",resize_img)

    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cam.release()
cv2.destroyAllWindows()

    




