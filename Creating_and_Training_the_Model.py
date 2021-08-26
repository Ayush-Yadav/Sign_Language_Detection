# Creating the model based on Convolutional Neural Networks (CNN)

import tensorflow
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
tensorflow.__version__

# Creating the required directory to store the Saved Model
if not os.path.exists('my_model'):
       os.makedirs('my_model')


# PART 1 - DATA PREPROCESSING

# Applying Data Augmentation to the Training Set Images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)

# Applying Data Augmentation to the Test Set Images
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training Set
training_set = train_datagen.flow_from_directory('my_dataset/training_set',
                                                  target_size = (200, 200),
                                                  batch_size = 128,
                                                  color_mode = "grayscale",
                                                  class_mode = 'categorical')

# Creating the Test Set
test_set = test_datagen.flow_from_directory('my_dataset/test_set',
                                             target_size = (200, 200),
                                             batch_size = 128,
                                             color_mode = "grayscale",
                                             class_mode = 'categorical')

# PART 2 - BUILDING THE CNN

# Initialising the CNN
model = Sequential()

## Following the VGGNet Architecture for CNN Model in case your dataset is quite huge

# Implementing the layers (Series of Convolution and Pooling Layers as is the combination in the VGGNet Architecture)
model.add(Conv2D(input_shape = (200, 200, 1),
                 filters = 16,
                 kernel_size = (3,3),
                 padding = "same",
                 activation = "relu"))
model.add(Conv2D(filters = 16,
                 kernel_size = (3,3),
                 padding = "same", 
                 activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),
                    strides = (2,2)))
model.add(Conv2D(filters = 32,
                 kernel_size = (3,3),
                 padding = "same", 
                 activation = "relu"))
model.add(Conv2D(filters = 32, 
                 kernel_size = (3,3), 
                 padding = "same", 
                 activation = "relu"))
model.add(Conv2D(filters = 32,
                 kernel_size = (3,3), 
                 padding = "same", 
                 activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2),
                    strides = (2,2)))
# model.add(Conv2D(filters = 64, 
#                  kernel_size = (3,3), 
#                  padding = "same", 
#                  activation = "relu"))
# model.add(Conv2D(filters = 64, 
#                  kernel_size = (3,3), 
#                  padding = "same", 
#                  activation = "relu"))
# model.add(Conv2D(filters = 64, 
#                  kernel_size = (3,3), 
#                  padding = "same", 
#                  activation = "relu"))
# model.add(MaxPool2D(pool_size = (2,2),
#                     strides = (2,2)))

# Flattening step
model.add(Flatten())

# Adding the Fully Connected layers

# Hidden Layer 1 with 64 Nodes/Neurons
model.add(Dense(units = 64))

# BatchNormalization Layer for NORMALIZING the input from previous layer 
model.add(BatchNormalization())

# Leaky Version of a Rectified Linear Unit as the Activation function
model.add(LeakyReLU(alpha = 0.2))

# Adding Dropout to the previous layer to prevent Overfitting
model.add(Dropout(rate = 0.5))

model.add(Dense(units = 64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(rate = 0.5))

# Adding the Output Layer according to our dataset
model.add(Dense(units = 28, 
                activation = 'softmax'))

# PART 3 - TRAINING THE CNN

# Compiling the CNN
opt = Adam(lr = 0.0001)

model.compile(optimizer = opt,                                                                             
              loss = 'categorical_crossentropy',                             
              metrics = ['accuracy'])

## Creating Useful Callbacks 

# Model Checkpoint to Save the Best model after each epoch
model_checkpoint = ModelCheckpoint('my_model/model.h5',
                                    monitor = "val_loss",
                                    verbose = 0,
                                    save_best_only = True)

# Reduce LR to prevent the model from Overshooting the minima of loss function            
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.2, 
                              patience = 1, 
                              min_lr = 0.0001)

# EarlyStopping algorithm so that if the Validation loss keeps on Increasing for 2 epochs then the Training is stopped                              
early_stop = EarlyStopping(monitor = 'val_loss', 
                           min_delta = 0, 
                           patience = 2, 
                           verbose = 0, 
                           mode = 'auto')

# Training the CNN on the Training set and validating it on the Test set
model.fit_generator(training_set,
                    steps_per_epoch = 218,
                    epochs = 10,
                    callbacks = [reduce_lr, early_stop, model_checkpoint],
                    validation_data = test_set,
                    validation_steps = 21)

## PART 4 - SAVING THE MODEL

## One can use the json file type to save it as follows :-
# model_json = model.to_json()
# with open("my_model/model.json", "w") as json_file:
#      json_file.write(model_json)
# model.save_weights('my_model/model.h5')

## Alternate Way of Saving :-
# model.save('my_model/model.h5')