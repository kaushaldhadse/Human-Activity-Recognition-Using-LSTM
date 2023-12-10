# Human Activity Recognition Using LSTM Deep Learning

## Introduction

This project gives a step-by-step explanation on how to build a Human activity Recognition (HAR) Model using Deep Learning. 

Human Activity Recognition is a topic on which wide research is going on. These activities are recognized using the sensor readings. The main goal of this project is to train a Deep Learning model using these readings for this task.

The dataset used here is UCI HAR Dataset. This dataset can be accessed [Here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The dataset contains signals from the Gyroscope and Accelerometer of a smartphone while doing six activities : WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

The dataset contains Time-Series data, therefore LSTM model was the best choice for this task.
