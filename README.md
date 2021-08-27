========== SIGN LANGUAGE DETECTION USING CONVOLUTIONAL NEURAL NETWORKS (CNNs) ==========
                                                                                                                      
* TECHNOLOGIES USED :-                                                                                             
                                                                                                                   
1. All the 4 files in the project are written entirely using Python.                                                
2. TensorFlow and keras libraries are used to handle everything related to the CNN model.                          
3. OpenCV is used to capture the images via webcam and to process those images before feeding them to the model.   
4. Visual Studio Code was used to write and execute all the codes in the project.                                  
                                                                                                                    
----------------------------------------------------------------------------------
                                                                                                                   
* BRIEF DESCRIPTION OF THE FILES :-                                                                                
                                                                                                                   
1. Data_Collection_and_Processing.py creates the Training and the Test (Validation) set for our CNN model via      
   webcam and processes it using OpenCV. It creates a separate directory for each label (Hand Sign) and puts the   
   relevant processed images in those directories.                                                                 
2. Hand_Segmentation.py detects hand in the Region of Interest (ROI) via Background Subtraction and Thresholding.  
3. Creating_and_Training_the_Model.py trains our CNN model on the Training set and evaluates it on the Test set.   
   I follow the VGGNet Architecture for the model. After training, the model is saved in a directory.              
4. Testing_the_model.py tests our CNN model by predicting labels for the Hand signs made live on the webcam. Run   
   this code directly if you want to see how my model predicts on new Hand images.                                                              
                                                                                                                   
----------------------------------------------------------------------------------
                                                                                                                   
* WHAT YOU CAN DO IN MY PROJECT :-                                                                                 
                                                                                                                   
1. Create your OWN Hand Signs for different labels like alphabets, numbers, etc. and Train my model on your Hand   
   signs and see how it performs.                                                                                  
2. Play with different methods in OpenCV to process the image (Hand Sign) according to your hand signs and the     
   quality of your webcam, some of those methods are commented out in my code for your reference.                  
3. Play with different parameters in my CNN model depending on the size of your dataset and see which architecture 
   suits your dataset the best, some more layers and nodes are commented out in my code for your reference again.  
4. I've added comments for each line of code in all the files so don't be afraid of looking into the code.         
__________________________________________________________________________________      
