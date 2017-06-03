#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imageATran]:  ./examples/after_trans.png	"Sample Image After Transformation"
[imageANorm]:  ./examples/aft_normal.png	"Sample Image After Normalization"
[imageAugTr]:  ./examples/aug_train.png		"Histogram of Augmented Training Data"
[imageBNorm]:  ./examples/bef_normal.png	"Before Normalization and Conversion to Grayscale"
[imageBTran]:  ./examples/before_trans.png	"Sample Image Before Transformation"
[imageiBump]:  ./examples/i_bumpy.png		"Internet Image Bumpy"
[imageiDang]:  ./examples/i_dangerous_left.png	"Internet Image Dangerous Left"
[imageiNarr]:  ./examples/i_narrows_right.png	"Internet Image Narrows Right"
[imageiSlip]:  ./examples/i_slippery.png	"Internet Image Slippery Road"
[imageiTraf]:  ./examples/i_traffic.png		"Internet Image Traffic"
[imageSamTr]:  ./examples/sample_train.png	"Sample Training Image"
[imageHiTes]:  ./examples/test_hist.png		"Test Data Histogram"
[imageAcTra]:  ./examples/train_acc.png		"Training and Validation Accuarcy Curves"
[imageHiTra]:  ./examples/train_hist.png	"Training Data Histogram Before Augmentation"
[imageLoTra]:  ./examples/train_loss.png	"Training and Validation Loss Curves"
[imageHiVal]:  ./examples/valid_hist.png	"Validation Data Histogram"


This document is a report on the project to design a convolutional neural network for the task of Traffic Sign Image Classification.
The code for this project is submitted along with this report. Please the directory named "code".
The code folder has the following files:
* TSC.ipynb -- The final python notebook used for the project.
* TSC.html -- Downloaded html of the python notebook.
* checkpoint and "lenet.\*" files -- These files contain the trained model.


The internet images downloaded are in the directory "german\_traffic". 
The versions of these scaled to 32x32 space are in the folder examples, and have names matching "i\_\*.png"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using the python library the basic statistics of the data set are summarized below.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32 x 32 x 3). The last 3 being the RGB color channels.
* The number of unique classes/labels in the data set is 43

This is in the section "Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas".

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

A sample image in the dataset is shown below.

![Sample Training Image][imageSamTr]

In order to understand the distribution of datasets with respect each class of traffic signs, histogram
plots for the training, validation and test data were plotted and are shown below.
![Training Data Set Histogram][imageHiTra]

![Validation Data Set Histogram][imageHiVal]

![Test Data Set Histogram][imageHiTes]

The main dataset of interest for the purpose of developing a neural network is of course the training
dataset. From the training dataset, it is clear that some classes have very little training data as
compared to others. In order for the neural network to train better, the training data was augmented using
affine transformations.

The code for this is in the section "Include an exploratory visualization of the dataset".

###Design and Test a Model Architecture

####1. Data Augmentation and Normalization

As described above, the training data had an imbalance in terms of the number of samples available
for each class. Therefore, I decided to generate additional data so that the classifier could be
trained better to identify new images and would not overfit the limited data in these classes.

In order to augment the training data, I transformed the original image using random affine transforms, and
also performed brightness adjustments. The affine transformations, namely, rotation, shear and translation compensate
for lack of camera images for a particular class being taken from different angles, distances and focus.
The brightness augmentation helps adjust images in the test set could potentially be taken in different lighting conditions.
Further, augmenting the training set using these parameters is a widely accepted practice.
In particular, I used random rotation with an angle range \[-10, 10\] degrees, random shear in the range \[-5, 5\],
and random x, y, translations in the range \[-2.5, 2.5\]. 

Here is an example of an original image and an augmented image:

![Original Image][imageBTran] ![Transformed Image][imageATran]

My functions to perform the low-level transformations were based on the wrappers around OpenCV utilities posted [here.](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.q69bj4ix7) 

In order to generate more data, I set a threshold of minimum of 2000 images per class and made sure I generated
extra images for each class to reach the threshold. The training data histogram after augmentation is shown below

![Augmented Training Data Histogram][imageAugTr]


The dataset provided to us was already split into training, validation, and test data sets, so I decided to use them.

My final training set had 86010 number of images. My validation set had 4410, and test set had 12630 images.


Finally, I decided to convert the images to grayscale. There were two main reasons for doing this, namely
intensity information is the main feature which we need for traffic sign classification, and, it simplifies
the model. I do understand there is a loss of information in this process, but for the purposes of this project
it is a reasonable tradeoff. One added benefit I observed was that for the internet images, the OpenCV resize
function modifies the color space, but despite this I got good accuracy on the internet images. This will
be described in sections below.

Here is an example of a traffic sign image before and after grayscaling.

![Before GrayScale][imageBNorm] ![After Grayscale AND normalization][imageANorm]

I normalized the data to be in the range -1.0, to +1.0, as this makes the optimizers perform better.

All of the above code is in the section "Pre-process the Data Set ".

####2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture (see the section "Model Architecture" in the notebook) described in the class. The only modification was to add dropouts
to activations to fully connected layers. My final model consisted of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 grayscale image, normalized   	| 
| Convolution      	| Input = 32x32x1, output=28x28x6  		|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  		|
| Convolution      	| Input = 14x14x6, output=10x10x16  		|
| RELU			| 						|
| Max pooling		| 2x2 stride, output=5x5x16			|
| Flatten		| Input=5x5x16, output=400			|
| Fully connected	| Input=400, output=120				|
| RELU			| with dropout					|
| Fully connected	| Input=120, output=84				|
| RELU			| with dropout					|
| Fully connected	| Input=84, output=43				|
| Softmax		|         					|


####3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for this is in the sections ("Train, Validate and Test the Model).
To train the model, I used Adams optimizer.
The dropouts parameter was set to 0.8, batch size was 128, and the number of epochs was 50.
The learning rate of 0.0006 worked well for me.


####4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for this is in the sections ("Train, Validate and Test the Model).
My final model results were:
* training set accuracy of   94.9
* validation set accuracy of 94.2
* test set accuracy of 	     92.5

The accuracy curves for training and the validation data with the epochs are shown below:

![Training and Validation Accuracy][imageAcTra]

The loss curves for training and the validation data with the epochs are shown below:

![Training and Validation Loss][imageLoTra]

If a well known architecture was chosen:
* What architecture was chosen?
  The LeNet architecture modified with dropouts in the RELU activations to the fully connected layer.
* Why did you believe it would be relevant to the traffic sign application?
  The LeNet architecture has shown excellent performance for the MNIST data to classify digits.
  The traffic sign detection problem is of a similar flavor, although potentially more complex,
  both with the space of traffic signs and also the number of class labels. However, with relatively
  minor modifications, the LeNet architecuture I believed that the network would generalize well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  With very little tuning, the architecture achieved accuracy of over 92% in all the data sets. This is very strong evidence
  that the network is working very well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Bumpy Road][imageiBump] ![Dangerous Left][imageiDang] ![Road Narrows Right][imageiNarr]
![Slippery Road][imageiSlip] ![Traffic Sign][imageiTraf] 

As seen from the above images, after resizing using opencv2 library function, 
the red color space does not seem to be handled properly by the resize method.
When I initially built a network to classify with RGB models, it did not get good accuracy
on these images. However, a grayscale transformation used in this report had less problems.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
The code for this is in the sections "Predict the Sign Type for Each Image", and the following "Analyze Performance".


Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road				| 
| Dangerous Left   		| Dangerous Left			|
| Road Narrows Right		| Pedestrians				|
| Slippery Road			| Slippery Road      			|
| Traffic Sign			| Traffic Sign				|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
This compares favorably to the accuracy on the test set of 92.5%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The softmax probabilities (see section "Output Top 5...") and the corresponding image predictions for the five traffic sign images are shown below:

Probabilities:
[[  9.99974847e-01   2.51656875e-05   5.02343589e-08   1.19079402e-13     1.78928272e-16]
 [  8.35657239e-01   1.64342403e-01   3.87548141e-07   2.71614975e-09     2.08691042e-09]
 [  5.27963400e-01   2.93871015e-01   7.91130215e-02   4.95926291e-02     3.76766622e-02]
 [  8.07768285e-01   1.91521749e-01   4.17222502e-04   1.86385005e-04     1.05500832e-04]
 [  6.86344624e-01   3.05126399e-01   7.74168689e-03   7.37974362e-04     4.56601811e-05]]

Predictions:
 [[22 26 18 29 27]
  [19 23 22 21 31]
  [27 26 18 24 22]
  [23 19 21 31 22]
  [26 18 22 27 23] ]

For example, for the first image, the model predicted correctly with a probability of 0.99, so it is very sure about the prediction.
The interpretation of the rest of the probalities and the corresponding classes are shown in the table below.
The probabilities
| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .99         		| Bumpy Road					| 
| .2-e04     		| Dangerous Left				|
| .5-e07		| Pedestrians					|
| .11-e13      		| Slippery Road					|
| .17-e16		| Traffic Sign      				|

One can generate a similar table for the other images.
The final predictions and the probabilites are summarized below.
Predicted Image Labels = [22 19 27 23 26]
Predicted Image Probs = [ 0.99997485  0.83565724  0.5279634   0.80776829  0.68634462]
The correct image labels were = [22, 19, 24, 23, 26]
As can be seen the model is unsure image with correct label 23, as probability is only 0.52.
Other than that it is reasonably sure about the other labels.


####4. Acknowledgement.

The code for LeNet was based on the code presented in the class.
The discussions on the forums were very helpful. I am also thankful to forum mentors for
pointing out the code for low level transformations [here,](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.q69bj4ix7) based on which I performed the data augmentation.

The 5 internet images were (as far as I can recollect) obtained from the wikipedia [link.](https://en.wikipedia.org/wiki/Road_signs_in_Germany)

