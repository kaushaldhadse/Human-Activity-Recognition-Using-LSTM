# Human Activity Recognition Using LSTM Deep Learning

## Introduction

This project gives a step-by-step explanation on how to build a **Human activity Recognition (HAR) Model** using Deep Learning. 

Human Activity Recognition is a topic on which wide research is going on. These activities are recognized using the sensor readings. The main goal of this project is to train a Deep Learning model using these readings for this task.

The dataset used here is **_UCI HAR Dataset_**. This dataset can be accessed [Here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The dataset contains signals from the Gyroscope and Accelerometer of a smartphone while doing six activities : **_WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING._**

For creating this dataset, experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded **_accelerometer_** and **_gyroscope_**, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

The dataset contains Time-Series data, therefore LSTM model was the best choice for this task.


## Data Preprocessing

The dataset contains ‘.txt’ files which contain signal data from the sensors.
There are a total nine types of signals in the dataset : **_"body_acc_x_","body_acc_y_","body_acc_z_","body_gyro_x_","body_gyro_y_","body_gyro_z_","total_acc_x_","total_acc_y_","total_acc_z_".-**

 These files were loaded in numpy arrays X_train, X_test, y_train and y_test. 
For this task, the functions ‘**_load_x-**’ and ‘**_load_y_**’were created.


## Model Description

The **_LSTM_** model is commonly utilized for Human Activity Recognition (HAR) due to its efficacy in handling time-series data. The model consists of four layers: **_LSTM Layer-1, Dropout Layer-1, LSTM Layer-2, Dropout Layer-2_**. 

The **_Adam optimizer_**, known for its efficiency in optimization, was employed. For multi-categorical classification, **_Softmax_** activation was utilized along with the **_"categorical_crossentropy"_** loss function, suitable for cases where the output variables (y_train, y_test) are one-hot encoded.

**_Dropout layers_** are strategically integrated to mitigate variance and prevent overfitting. After the model was compiled, various combinations of **_'epochs'_** and **_'batch_size'_** were experimented with to achieve the desired accuracy.


## Result

The model reached an impressive final accuracy of **_91.48%_**. This was achieved with a batch size of 32 and 20 epochs. 

To tackle overfitting, a **_dropout probability_** of 0.7 was maintained in both dropout layers, effectively reducing variance.
