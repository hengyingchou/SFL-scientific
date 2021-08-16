import cv2
import numpy as np
import time

import os
from datetime import datetime

# save image path
path = '/home/cielsun/Desktop/yolo/Alexey_darknet/darknet-master/ImageFiles5'

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","cfg/yolov3.cfg")

# Using cuda to process
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# using coco label names for object detection
classes = []
with open("data/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))
# Assign the camera we want to use
cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()


## Put object detection into function
def detect():

    #Reveive the image
    _,frame= cap.read()
    
    # Cropping the image
    frame = np.array(frame)
    frame = frame[430:850,750:1450,::]
    height,width,channels = frame.shape

    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    
    net.setInput(blob)
    outs = net.forward(outputlayers)

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # only get the object with confidence rate higher than 30%
            # store object's location, size of bounding box
            if confidence > 0.3:
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                

                boxes.append([x,y,w,h]) 			  #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) 			  #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


    # Put the label, bounding box and cofidence rate on the picture
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)


    # put number of frames on the picture
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)

	
    # saving file
    # using date and time as saving name
    date = datetime.now()
    current_date = str(date)
    print(current_date)
    cv2.imwrite(os.path.join(path,current_date + ".jpg"),frame)

if __name__ == '__main__':
    
    frame_id = 0
    # get the initial time
    time1 = time.time()
    while True:

    	# get the time
        temp = time.time()
        #do object detection every 2 minutes
        if(temp - time1 > 120):
            time1 = temp
            detect()

        #wait 1ms the loop will start again and we will process the next frame
        key = cv2.waitKey(1) 
        #esc key stops the process
        if key == 27: 
            break
    
    cap.release()    
    cv2.destroyAllWindows()
