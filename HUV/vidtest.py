import cv2

def vid():
    cap= cv2.VideoCapture(0)

    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Specify save directories
    writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height)) 


    while True:
        ret,frame= cap.read()

        writer.write(frame)

        cv2.imshow('frame', frame)

        #Esc break key
        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    writer.release()
    cv2.destroyAllWindows()


