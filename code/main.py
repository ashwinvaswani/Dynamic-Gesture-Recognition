from pynput.keyboard import Key,Controller
import os

import numpy as np
import cv2
import time
from keras.models import Model, load_model
import keyboard
import pyautogui

def minimizer():
    # time.sleep(0.2)
    try:     
        keyboard.press_and_release('cmd+d')
    except Exception as e:
        return str(e)
    return "success"

def tabs_cycle():
    try:
        keyboard.press_and_release('alt+shift+tab')
    except Exception as e:
        return str(e)
    return "success"

def bring_up_tabs():
#     time.sleep(3)
    try:
        keyboard.press_and_release('win+shift+m')
    except Exception as e:
        return str(e)
    return "success"

def bring_down_tabs():
#     time.sleep(0.5)
    try:
        keyboard.press_and_release('win+m')
    except Exception as e:
        return str(e)
    return "success"

def take_ss(count=[0]):
    count[0]=count[0]+1
    myss = pyautogui.screenshot()
    myss.save('./Screenshots/ss'+str(count[0])+'.png')


dict_ind_to_class = {0:'Pulling Hand In',2:'Swipe Left',1:'Swipe Right',3:'Thumb Up',4:'No Gesture'}

model1_path = "../Models/Model_84_5_jester"
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
instruction = 'No Gestrue'
pre =0
prev2 = None
prev = None

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
        frames = frames[4:]
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
        if num[0]==prev and prev!=prev2:
            if num[0]==0:
                print(minimizer())
            if num[0]==1:
                print(bring_down_tabs())
            if num[0]==2:
                print(tabs_cycle())
            if num[0]==3:
                print(bring_up_tabs())
        print("Curr Gesture : ",dict_ind_to_class[num[0]],"prev :",dict_ind_to_class[prev] if prev!=None else '000',"prev2 :",dict_ind_to_class[prev2] if prev2!=None else '-----')
        prev2 = prev
        prev = num[0]
        
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
    




