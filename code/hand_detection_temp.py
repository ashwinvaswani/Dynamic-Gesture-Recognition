import cv2
import datetime
import os
import argparse
import imutils
from imutils.video import VideoStream

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from keras.models import load_model
import keras
from keras.applications import ResNet50,VGG16
from keras.applications.resnet50 import preprocess_input
from keras import Model,layers
from keras.models import load_model,model_from_json

from utils import detector_utils as detector_utils

import time
import subprocess 


# Initialising dictionary
dict_ind_to_class = {0:'Pulling Hand In',1:'Swipe Right',2:'Swipe Left',3:'Thumb Up',4:'No Gesture'}


def get_labels_for_plot(predictions):
    predictions_labels = []
    for ins in labels_dict:
        if predictions == labels_dict[ins]:
            predictions_labels.append(ins)
            return predictions_labels
            #break
    #return predictions_labels

def load_test_dataa():
    images = []
    names = []
    size = 64,64
    temp = cv2.imread('./img2.jpg')
    temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
    cv2.imwrite('out.jpg',temp)
    temp = cv2.resize(temp, size)
    images.append(temp)
    names.append('img2')
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names

def minimizer():
    #For Windows
    if os.name == 'nt':
        import keyboard
        try:     
            keyboard.press_and_release('win+shift+down') 
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:     
            subprocess.call(["xte","keydown Super_L","key h","keyup Super_L"]) 
        except Exception as e:
            return str(e)
        return "success"


def tabs_cycle():
    #For Windows
    if os.name == 'nt':
        import keyboard
        try:     
            keyboard.press_and_release('alt+shift+tab')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Alt_L","keydown Shift_L","key Tab","keyup Shift_L","keyup Alt_L"]) 
        except Exception as e:
            return str(e)
        return "success"

def bring_up_tabs():
    #For Windows
    if os.name == 'nt':
        import keyboard
        try:     
            keyboard.press_and_release('win+d')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Super_L","keydown Shift_L","key m","keyup Shift_L","keyup Super_L"])
        except Exception as e:
            return str(e)
        return "success"

def bring_down_tabs():
    #For Windows
    if os.name == 'nt':
        import keyboard
        try:     
            keyboard.press_and_release('win+m')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Super_L","key d","keyup Super_L"])
        except Exception as e:
            return str(e)
        return "success"

def take_ss(count=[0]):
    #For Windows
    if os.name == 'nt':
        import pyautogui
        count[0]=count[0]+1
        try:     
            keyboard.press_and_release() 
        except Exception as e:
            return str(e)
        return "success"
    else:
        count[0]=count[0]+1
        try:
            # keyboard.press_and_release(["xte","keydown Super_L","key d","keyup Super_L"])
            path = './Screenshots/ss'+str(count[0])+'.png' 
            subprocess.call(["xte","scrot",path])
        except Exception as e:
            return str(e)
        return "success"

def refresher():
    # For windows
    if os.name == 'nt':
        import keyboard
        try:
            keyboard.press_and_release('ctrl+r')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Ctrl_L","key r","keyup Ctrl_L"])
        except Exception as e:
            return str(e)
        return "success"

def vlc_right():
    # For windows
    if os.name == 'nt':
        import keyboard
        try:
            keyboard.press_and_release('ctrl+right')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Ctrl_L","key right","keyup Ctrl_L"])
        except Exception as e:
            return str(e)
        return "success"

def vlc_left():
    # For windows
    if os.name == 'nt':
        import keyboard
        try:
            keyboard.press_and_release('ctrl+right')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Ctrl_L","key left","keyup Ctrl_L"])
        except Exception as e:
            return str(e)
        return "success"


def vlc_volumeup():
    # For windows
    if os.name == 'nt':
        import keyboard
        try:
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
            keyboard.press_and_release('ctrl+up')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
            subprocess.call(["xte","keydown Ctrl_L","key up","keyup Ctrl_L"])
        except Exception as e:
            return str(e)
        return "success"

def vlc_pause():
    # For windows
    if os.name == 'nt':
        import keyboard
        try:
            keyboard.press_and_release('space')
        except Exception as e:
            return str(e)
        return "success"
    else:
        try:
            subprocess.call(["xte","space"])
        except Exception as e:
            return str(e)
        return "success"


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    #   Loading model
    model1_path = "../Models/Model_84_5_jester"
    model = load_model(model1_path)

    font = cv2.FONT_HERSHEY_SIMPLEX
    quietMode = False
    img_rows,img_cols=64, 64 

    # Get stream from webcam and set parameters)
    vs = VideoStream().start()

    # max number of hands we want to detect/track
    num_hands_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)


    # Initialising varibales
    framecount = 0
    detect_count = 0
    fps = ""
    start = time.time()
    frames = []
    num=[5]
    max =1
    real_index = 5
    instruction = 'No Gesture'
    pre =0
    prev = None
    prev2 = None
    prev3 = None
    black = np.zeros((100, 400, 3), dtype = "uint8")
    num_classes = 5

    arr = []
    final_arr = []
    counter = 0
    print("Enter first gesture")

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original',720,720)

    try:
        while True:
            # Read Frame and process
            frame = vs.read()

            
            frame = cv2.flip(frame, 3)
            frame = cv2.resize(frame, (720,720))

            frame_copy = frame.copy()

            if im_height == None:
                im_height, im_width = frame.shape[:2]


            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX 

            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            #print(boxes)


            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # print(type(scores))

            if (np.ndarray.max(scores) < 0.6):
                # print("no hand detected")
                detect_count -= 1
                if detect_count < 0:
                    detect_count = 0
            else:
                # print("Hand detected")
                detect_count += 1
                if detect_count >16:
                    detect_count = 16

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
                frames = frames[2:]
                X_tr.append(input)
                X_train= np.array(X_tr)
                train_set = np.zeros((1, 16, img_cols,img_rows,3))
                train_set[0][:][:][:][:]=X_train[0,:,:,:,:]
                train_set = train_set.astype('float32')
                train_set -= 108.26149
                train_set /= 146.73851
                result_1 = model.predict(train_set)
                # print(result_1)
                num = np.argmax(result_1,axis =1)
                instruction = dict_ind_to_class[num[0]]
                print(detect_count)
                if num[0]==prev and prev!=prev2 and detect_count > 0 and prev!= prev3:
                    if num[0]==0:
                        print(bring_down_tabs())
                    if num[0]==1:
                        print(minimizer())
                    if num[0]==2:
                        print(tabs_cycle())
                    if num[0]==3:
                        print(take_ss())
                # print("Curr Gesture : ",dict_ind_to_class[num[0]],"prev :",dict_ind_to_class[prev] if prev!=None else '000',"prev2 :",dict_ind_to_class[prev2] if prev2!=None else '-----')
                prev3 = prev2
                prev2 = prev
                prev = num[0]
                
            cv2.putText(frame, instruction, (450, 50), font, 0.7, (0, 255, 0), 2, 1)
            cv2.putText(black, "Quiet Mode  "+instruction, (0,50), font, 0.8, (255, 255, 255), 2, 1)
            if not quietMode:
                cv2.resizeWindow('Original',720,720)
                cv2.imshow('Original',frame_copy)
            if quietMode:
                cv2.resizeWindow('Original',100,400)
                cv2.imshow('Original',black)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('q'):
                quietMode = not quietMode
                # minimizer()
        cap.release()
        cv2.destroyAllWindows()





            



            # # Calculate Frames per second (FPS)
            # num_frames += 1
            # elapsed_time = (datetime.datetime.now() -
            #                 start_time).total_seconds()
            # fps = num_frames / elapsed_time

            # if args['display']:
            #     detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            #     cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            #     if cv2.waitKey(25) & 0xFF == ord('q'):
            #         cv2.destroyAllWindows()
            #         vs.stop()
            #         break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))