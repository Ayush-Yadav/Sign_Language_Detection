import cv2

# Below function extracts the hand from the background in a given frame based on Background Subtraction and then returns it after 
# Thresholding so as to get a better estimation of the hand shape and orientation  
def segment_hand(frame, background, threshold=25):
    
    # Background Subtraction on the current frame
    diff = cv2.absdiff(src1 = background.astype("uint8"),
                       src2 = frame)

    # Creating a normal Thresholding for the contours to detect if there is a hand or not
    _, normThresh = cv2.threshold(src = diff,
                                  thresh = threshold,
                                  maxval = 255,
                                  type = cv2.THRESH_BINARY)

    # Thresholding can be done depending on the physical environment you're in and also what mainly differentiates your hand signs
    ## thresholded = cv2.Canny(diff, 100, 200)                           // Use it without performing further thresholding if your hand 
    #                                                                       signs only depend on the outline of the hand
    ## threshold1 = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)    // Mean Adaptive thresholding can 
    #                                                                                                    also be used instead of Gaussian
    #                                                                                                    depending on the above factors
    threshold1 = cv2.adaptiveThreshold(src = diff,
                                       maxValue = 255, 
                                       adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType = cv2.THRESH_BINARY,
                                       blockSize = 11,
                                       C = 2)
    _ , thresholded = cv2.threshold(src = threshold1,
                                    thresh = threshold,
                                    maxval = 255,
                                    type = cv2.THRESH_BINARY)
    # Adaptive Thresholding can also be dropped depending on the situation 

    # Grab the external contours for the thresholded image
    contours, hierarchy = cv2.findContours(image = normThresh.copy(),
                                           mode = cv2.RETR_EXTERNAL,
                                           method = cv2.CHAIN_APPROX_SIMPLE)
    
    # Return the hand if its contour is found else return None 
    if len(contours) == 0:

        return None
    else:

        hand_segment_max_cont = max(contours, 
                                    key = cv2.contourArea)

        return (thresholded, hand_segment_max_cont)