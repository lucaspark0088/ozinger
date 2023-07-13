from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Fallback for TensorFlow 1.x.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # TensorFlow 2.x

tf.get_logger().setLevel('ERROR')

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

a = 0
b = 0

prev_label = ""
current_label = ""

##############################################
# 디바이스전환- 디바이스모드
# 명령전환- 명령모드 

Mode = "device_mode"  # cmd_mode
Cmd = ""
while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    a +=1

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    flipped = cv2.flip(image, 1)
    # Show the image in a window
    cv2.imshow("Webcam Image", flipped)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)


    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    if Mode == "device_mode":
        if class_name[2:-1] == 'select_light':
            print(prediction)
            if prediction[0] * 100 >= 70:
                print(class_name[2:])
            # print(prediction)
                Mode = "cmd_mode1"     
        elif class_name[2:-1] == "select_motor":
            if prediction * 100 >= 70:
                print(class_name[2:])
                Mode = "cmd_mode2"   

    elif Mode == "cmd_mode1":
        if class_name[2:-1] == 'turn_on':
            if prediction[0] * 100 >= 70:
                print(class_name[2:])
            # for bb in range(500):
            #     b = b + bb
            #     b = 0
            Mode = "device_mode"

        elif class_name[2:-1] == 'turn_off':
            if prediction[0] * 100 >= 70:
                print(class_name[2:])
            # for bb in range(500):
            #     b = b + bb
            #     b = 0
            Mode = "device_mode"
            
    
    elif Mode == "cmd_mode2":
        if class_name[2:-1] == 'power_up':
            if prediction[0] * 100 >= 70:
                print(class_name[2:])
                # for bb in range(500):
                #     b = b + bb
                #     b = 0
                Mode = "device_mode"

        elif class_name[2:-1] == 'power_down':
            if prediction[0] * 100 >= 70:
                print(class_name[2:])
                # for bb in range(500):
                #     b = b + bb
                #     b = 0
                Mode = "device_mode"


    # if class_name[2:-1] == 'select_light':
    #     print(class_name[2:])
    
    # elif class_name[2:-1] == 'turn_on':
    #         print(class_name[2:])

        


    # if a == 10:
    #     print(class_name[2:], end="")
    #     a = 0

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
