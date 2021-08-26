import cv2
import time
import os
from string import ascii_uppercase
from Hand_Segmentation import segment_hand

# Creating Relevant folders to store the data collected for Hand Signs (One can create a new folder by changing the below variable)
my_folder_path = 'my_dataset'

if not os.path.exists(my_folder_path):
       os.makedirs(my_folder_path)
if not os.path.exists(my_folder_path + '/training_set'):
       os.makedirs(my_folder_path + '/training_set')
if not os.path.exists(my_folder_path + '/test_set'):
       os.makedirs(my_folder_path + '/test_set')

# Initialize the 'running-averaged' Hand Background for future Background Subtraction  
## MAKE SURE YOUR HAND BACKGROUND IS AS PLAIN AS POSSIBLE AND NOT TOO BRIGHT (FOR BETTER EXTRACTION OF THE HAND FEATURES FROM THE FRAME)
background = None
alpha_val = 0.5

# Defining dimensions for the Region of Interest (ROI) where the Hand should be placed
ROI_top = 100
ROI_bottom = 350
ROI_right = 200
ROI_left = 440

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

# Opening the camera for Video Capturing Hand Gestures/Signs
cam = cv2.VideoCapture(0)

# Labels for My dataset [ '0' --> Backspace, '1' --> Space, Rest are the 26 English Alphabets]
labels = ['0', '1']     
for char in ascii_uppercase:

    labels.append(char)

# Generating and Storing the Dataset for each Label one-by-one
for label in labels:

   k = 0                                                            # Press Keys N to 'skip to the next Label' and ESC to 'exit the cam'

   # Resetting these global variables for each Label 
   background = None
   alpha_val = 0.5

   num_frames_taken = 0
   num_imgs_taken = 0
   
   # Creating a folder/directory for each Label in the Training as well as in the Test Set
   if not os.path.exists(my_folder_path + '/training_set/' + str(label)):
          os.makedirs(my_folder_path + '/training_set/' + str(label))
   if not os.path.exists(my_folder_path + '/test_set/' + str(label)):
          os.makedirs(my_folder_path + '/test_set/' + str(label))
   
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

            # Informing the User that Background Calculation is going on...
            if num_frames_taken <= 99:

               cv2.putText(img = frame_copy,
                           text = "FETCHING BACKGROUND...PLEASE WAIT!",
                           org = (20, 400),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (0,0,255),
                           thickness = 2)

               # cv2.imshow("Sign Detection", frame_copy)

        elif num_frames_taken <= 200: 

            # Segmenting and Adjusting the Hand properly for the next 100 ROI frames
            hand = segment_hand(gray_frame, background)
            
            # Informing the User to get ready and make Hand Gestures/Signs for the Current Label
            cv2.putText(img = frame_copy,
                        text = "Adjust Hand Gesture for " + str(label),
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
               
               # Informing the User about the number of frames taken till now
               cv2.putText(img = frame_copy,
                           text = str(num_frames_taken - 100) + " frames taken for adjusting " + str(label),
                           org = (70, 70),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (0,0,255),
                           thickness = 2)

               # Also display the thresholded image
               cv2.imshow(winname = "Thresholded Hand Image",
                          mat = thresholded)

        else: 

            ## Filling up the dataset with appropriate Hand Images/Signs

            # Segmenting the Hand region
            hand = segment_hand(gray_frame, background)
        
            # Checking if we are able to detect the Hand...
            if hand is not None:

                # Unpacking the thresholded image and the max_contour
                thresholded, hand_segment = hand

                # Drawing Contours around the Hand segment
                cv2.drawContours(image = frame_copy,
                                 contours = [hand_segment + (ROI_right, ROI_top)],
                                 contourIdx = -1,
                                 color = (255,0,0),
                                 thickness = 1)
                
                # Informing the User about the number of frames taken till now
                cv2.putText(img = frame_copy,
                            text = "Total " + str(num_frames_taken) + " frames taken for this Label yet",
                            org = (70, 70),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 1,
                            color = (0,0,255),
                            thickness = 2)

                # Displaying the thresholded image
                cv2.imshow(winname = "Thresholded Hand Image",
                           mat = thresholded)
                
                # Saving first 1000 Thresholded Images to the Training Set and next 100 to the Test Set
                if num_imgs_taken < 1000:

                   # Updating the User about the Current Stats 
                   cv2.putText(img = frame_copy,
                               text = str(num_imgs_taken) + " images taken of " + str(label) + " for training set",
                               org = (5, 400),
                               fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale = 1,
                               color = (0,0,255),
                               thickness = 2)

                   # Saving the Image to the Corresponding folder
                   cv2.imwrite(filename = my_folder_path + '/training_set/' + str(label) + '/' + str(num_imgs_taken) + '.jpg',
                               img = thresholded)

                elif num_imgs_taken < 1100:

                   # Updating the User about the Current Stats 
                   cv2.putText(img = frame_copy,
                               text = str(num_imgs_taken - 1000) + " images taken of " + str(label) + " for test set",
                               org = (5, 400),
                               fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale = 1,
                               color = (0,0,255),
                               thickness = 2)

                   # Saving the Image to the Corresponding folder           
                   cv2.imwrite(filename = my_folder_path + '/test_set/' + str(label) + '/' + str(num_imgs_taken - 1000) + '.jpg',
                               img = thresholded)

                else:

                   break

                num_imgs_taken += 1

                # Indicating the User that the next batch should be for the Test Set 
                if num_imgs_taken == 1000:

                   time.sleep(2)

            else:

               # Informing the User that there's no Hand detected 
               cv2.putText(img = frame_copy,
                           text = "No hand detected...",
                           org = (80, 400),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = (0,0,255),
                           thickness = 2) 

        # Drawing ROI on the Frame copy for User Convenience
        cv2.rectangle(img = frame_copy,
                      pt1 = (ROI_left, ROI_top),
                      pt2 = (ROI_right, ROI_bottom),
                      color = (255,128,0),
                      thickness = 3)
        
        # Project Motive
        cv2.putText(img = frame_copy,
                    text = "Hand Sign Recognition!!!",
                    org = (10, 20),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1,
                    color = (0,255,0),
                    thickness = 2)
    
        # Incrementing the number of frames for tracking
        num_frames_taken += 1

        # Display the frame with segmented Hand
        cv2.imshow(winname = "Sign Detection",
                   mat = frame_copy)

        # Closing the Current Window with the ESC key and the Current Label with N key
        k = cv2.waitKey(delay = 1)

        if k == ord('n') or k & 0xff == 27:

            break
   
   if k & 0xff == 27:

       break
   
   # Time delay on moving to the next Label
   time.sleep(5)

# Releasing the Camera & Destroying all the windows
cam.release()
cv2.destroyAllWindows()