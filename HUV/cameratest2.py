import cv2

def cam():
    
    #The '0' selects the first camera available
    vid = cv2.VideoCapture(0)

    while(True):
        _, frame = vid.read()
        cv2.imshow('camera', frame)
        
        #Break key
        if cv2.waitKey(1) & 0xFF == ord('j'):
            break
    
    vid.release()
    cv2.destroyAllWindows()
    
    return

while(True):
    cam()