import urllib.request
from deepface import DeepFace
import cv2
import imutils
import numpy as np

url='IP_WEBCAM_URL/IMAGE.jpg'

while True:
    imgpath=urllib.request.urlopen(url)

    
    '''imgpath.read() reads the uploaded image as binary data.
       bytearray(...) turns it into a mutable byte array.
       np.array(..., dtype=np.uint8) converts it into a NumPy array of unsigned 8-bit integers (this is the standard format for image data).'''
    imgnp=np.array(bytearray(imgpath.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    frame=imutils.resize(frame,width=450)

    
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print("Detection error:", e)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
