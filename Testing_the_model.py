import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import cv2
from Hand_Segmentation import segment_hand
from string import ascii_uppercase
import os
import time
tensorflow.__version__

# Creating a current Directory to store the Images for prediction 
if not os.path.exists('my_dataset/cur_data'):
       os.makedirs('my_dataset/cur_data')

if not os.path.exists('my_dataset/cur_data/0'):
       os.makedirs('my_dataset/cur_data/0')

# Loading our CNN Model
model = load_model('my_model/model.h5')

# Initialize the 'running-averaged' Hand Background for future Background Subtraction  
## MAKE SURE YOUR HAND BACKGROUND IS AS PLAIN AS POSSIBLE AND NOT TOO BRIGHT (FOR BETTER EXTRACTION OF THE HAND FEATURES FROM THE FRAME)
## It will be best if the Background is same as that of the Data Collection step, though not necessary 
background = None
alpha_val = 0.5

# Defining dimensions for the Region of Interest (ROI) where the Hand should be placed
ROI_top = 100
ROI_bottom = 350
ROI_right = 200
ROI_left = 440

# Creating the Dictionary for Labels
word_dict = {0: 'backspace', 1: 'space'}
i = 2
for char in ascii_uppercase:

    word_dict[i] = char
    i = i + 1

# Function to calculate the 'running-averaged' Background of the ROI
def cal_accum_avg(frame, alpha_val):

    global background
    
    # Background must not be empty for the 'accumulateWeighted' method of OpenCV
    if background is None:

        background = frame.copy().astype("float")
        return None
    
    # OpenCV method to calculate the Running Average Background
    cv2.accumulateWeighted(src = frame,
                           dst = background,
                           alpha = alpha_val)

# Opening the Camera for Video Capturing Hand Gestures/Signs
cam = cv2.VideoCapture(0)

num_frames_taken = 0                 
prev_char = ''
imag = None                                                                         # To store the Current Thresholded Image (Hand Sign)

# Function to Predict the class of the Current Hand Sign using our Pre-Trained Model
def make_prediction():

    global model
    global prev_char
    
    # Rescaling the Current Hand Sign to match our Pre-Trained Model's Image Specs
    curr_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    
    # Getting the Rescaled Image from the current Directory
    curr_img = curr_datagen.flow_from_directory('my_dataset/cur_data',
                                                 target_size = (200, 200),
                                                 batch_size = 1,
                                                 color_mode = "grayscale", 
                                                 class_mode = 'categorical', 
                                                 shuffle = False)
    
    # Predicting the Probabilities of all the classes for the Current Hand Sign 
    class_pred = model.predict(curr_img)

    # Storing the Highest Probability Class in 'prev_char' variable
    prev_char = word_dict[np.argmax(class_pred[0])]

while True:

    # Capturing the Current Frame
    ret, frame = cam.read()

    # Flipping the frame to prevent Inverted image of the Captured frame
    frame = cv2.flip(src = frame,
                     flipCode = 1)

    frame_copy = frame.copy()

    # Extracting our Region of Interest (ROI) from the Frame
    roi = frame[ROI_top: ROI_bottom, ROI_right: ROI_left]

    # Applying Gaussian Blur to the ROI after converting it to Grayscale (Since color of the Hand isn't relevant for our cause)                                                                                                 
    gray_frame = cv2.cvtColor(src = roi, 
                              code = cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(src = gray_frame,
                                  ksize =  (9, 9),
                                  sigmaX = 0)

    if num_frames_taken < 100:

        # Calculating the 'running-averaged' Background for the first 100 ROI frames
        cal_accum_avg(gray_frame, alpha_val)
        
        if num_frames_taken <= 99:

            # Displaying the Predicted Class for the Previous Hand Sign 
            cv2.putText(img = frame_copy,
                        text = "Previous Character :- " + str(prev_char),
                        org = (70, 70),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (0,0,255),
                        thickness = 2)

            # Informing the User that Background Calculation is going on... 
            cv2.putText(img = frame_copy,
                        text = "FETCHING BACKGROUND...PLEASE WAIT",
                        org = (20, 400),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (0,0,255),
                        thickness = 2)

            # cv2.imshow("Sign Detection", frame_copy)
    
    elif num_frames_taken <= 200:

        # Segmenting and Adjusting the Hand properly for the next 100 ROI frames
        hand = segment_hand(gray_frame, background)
            
        # Informing the User to get ready and make the Hand Gesture/Sign to be predicted
        cv2.putText(img = frame_copy,
                    text = "Adjusting Hand Gesture/Sign",
                    org = (80, 400),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (0,0,255),
                    thickness = 2)
        
        # Checking if Hand is actually there or not by Counting the number of Contours detected
        if hand is not None:

            # Unpacking the thresholded image and the max_contour               
            thresholded, hand_segment = hand

            # Drawing Contours around the Hand segment
            cv2.drawContours(image = frame_copy,
                             contours = [hand_segment + (ROI_right, ROI_top)],
                             contourIdx = -1,
                             color = (255,0,0),
                             thickness = 1)
               
            # Informing the User about the number of frames taken till now for Adjusting the Hand Gesture/Sign
            cv2.putText(img = frame_copy,
                        text = str(num_frames_taken - 100) + " frames taken for Adjusting Hand",
                        org = (25, 70),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = (0,0,255),
                        thickness = 2)

            # Also display the thresholded image
            cv2.imshow(winname = "Thresholded Hand Image",
                       mat = thresholded)
            
            # Storing the final Adjusted Hand Sign in 'imag'
            if num_frames_taken == 200:

                imag = thresholded

        else:

            num_frames_taken = 99

    else:

        # Saving the Hand Image/Sign to the current Directory
        cv2.imwrite(filename = 'my_dataset/cur_data/0/1.jpg',
                    img = imag)

        # Predicting the label/class for the Current Frame's Hand Gesture using our Pre-Trained Model
        make_prediction()

        # Resetting the 'num_frames_taken' to repeat the process 
        num_frames_taken = 0

        time.sleep(3)

    # Draw ROI on the Frame_copy for User Convenience
    cv2.rectangle(img = frame_copy,
                  pt1 = (ROI_left, ROI_top),
                  pt2 = (ROI_right, ROI_bottom),
                  color = (255,128,0),
                  thickness = 3)

    # Incrementing the number of frames for tracking
    num_frames_taken += 1

    # Project Motive
    cv2.putText(img = frame_copy,
                text = "Hand Sign Recognition!!!",
                org = (10, 20),
                fontFace = cv2.FONT_ITALIC,
                fontScale = 1,
                color = (0,255,0),
                thickness = 2)

    # Display the frame with segmented Hand
    cv2.imshow(winname = "Sign Detection",
               mat = frame_copy)

    # Close Windows with the ESC Key
    k = cv2.waitKey(delay = 1) & 0xff

    if k == 27:

        break

# Releasing the Camera and Destroying all the windows
cam.release()
cv2.destroyAllWindows()
