# DetectingCars
Algorithm used: RCNN

Dataset: Images with corresponding LiDar data and bounding boxes coordinates, Cifar-10 dataset.
The system was set up with MATLAB and CUDA accelerators on Window 10 for training the
Convolutional Neural Networks in this project.

All the images were down-sampled and grey scaled for both training and testing(attached
extractLearningData.m file which generates the modified training dataset). The bounding boxes were
checked for validity to ensure it is visible in the image, not occluded too much, close enough, small
enough and doesn’t belong to any class labels such as Unknown, Motorcycles, Industrial, Cycles, Boats,
Helicopters, Planes, Commercial, Trains. This was done before training the CNN.
There are two parts in R-CNN algorithm:

1. Training a basic CNN to detect objects in the images (this is our primary classifier that says
whether a given regional proposal contains a car or not).
Cifar-10 dataset was downloaded and used for training this CNN.

2. Obtaining the region proposal network (RPN) for generating region proposals and training the RCNN.
The R-CNN was trained on around 2100 training images from the dataset given as part of this challenge.
(we did this as a lot of images were repetitive and to save computation time).

The regional proposal network ranks region boxes and proposes the region most likely containing the
objects. Each of these regions are tested on the trained CNN to determine whether the region contains the
cars.
We repeat this for all the test images and obtain the number of cars in the image along with ground truth bounding boxes.
