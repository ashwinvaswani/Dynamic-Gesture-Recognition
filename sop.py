from  pynput import mouse, keyboard
from pynput.keyboard import Key
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from keras.models import Model, load_model

dict_ind_to_class = {0:'Pulling Hand In',2:'Swipe Left',1:'Swipe Right',3:'Thumb Up',4:'No Gesture'}

model1_path = "./3DCNN_LRN_112_6_jester"
model1 = load_model(model1_path)

font = cv2.FONT_HERSHEY_SIMPLEX
quietMode = False
img_rows,img_cols=64, 64 
cap = cv2.VideoCapture(0)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original',720,720)

# set rt size as 640x480
# cap.set(3, 1280)
# cap.set(4, 720)
framecount = 0
fps = ""
start = time.time()
frames = []
num=[5]
max =1
real_index = 5
instruction = 'no Gestrue'
pre =0

num_classes = 5
while(1):
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))
    

    framecount = framecount + 1
    end  = time.time()
    timediff = (end - start)
    if( timediff >= 1):
        fps = 'FPS:%s' %(framecount)
        start = time.time()
        framecount = 0

    cv2.putText(frame,fps,(10,20), font, 0.7,(0,255,0),2,1)
    X_tr=[]
         
    image=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frames.append(gray)
    input=np.array(frames)
    
    if input.shape[0]==16:
        frames = []
        X_tr.append(input)
        X_train= np.array(X_tr)
        train_set = np.zeros((1, 16, img_cols,img_rows,3))
        train_set[0][:][:][:][:]=X_train[0,:,:,:,:]
        train_set = train_set.astype('float32')
        train_set -= 108.26149
        train_set /= 146.73851
        result_1 = model1.predict(train_set)
        # print(result_1)
        num = np.argmax(result_1,axis =1)
        instruction = dict_ind_to_class[num[0]]
        print(dict_ind_to_class[num[0]])
        
    cv2.putText(frame, instruction, (450, 50), font, 0.7, (0, 255, 0), 2, 1)
    if not quietMode:
            cv2.imshow('Original',frame)
    key = cv2.waitKey(1) & 0xFF
    ## Use Esc key to close the program
    if key == 27:
        break
    elif key == ord('q'):
        quietMode = not quietMode
        print("Quiet Mode - {}".format(quietMode))
cap.release()
cv2.destroyAllWindows()
    




