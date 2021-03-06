**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset-exploratory-visualzn.jpg "Visualization"
[image2]: ./examples/30_limit.jpg "Before Normalized"
[image3]: ./examples/30_limit_after_normalization.jpg "Normalized"
[image4]: ./examples/no_entry.png "Traffic Sign 1"
[image5]: ./examples/priority_road.jpg "Traffic Sign 2"
[image6]: ./examples/right_ahead.jpg "Traffic Sign 3"
[image7]: ./examples/road_narr_right.jpg "Traffic Sign 4"
[image8]: ./examples/stop.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Here is a link to my [project code](https://github.com/suryakn/CARND-trafficsignclafr/blob/master/Traffic_Sign_Classifier.ipynb)


####Data Set Summary & Exploration

####1. I used the pickle library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. I have started without pre-processing and got only 83% validation accuracy.
And later I tried grayscale with no luck and then
I normalized the data which gave me good accuracies upto 98%

As a last step, I normalized the image using range normalization with range in -.5 to .5 as the data is from real life and properties like brightness, colors vary vastly and normalization shrinks the variation into a small range so that the weights/biases can be applicable to all the images.

Here is an image before and after normalization

![alt text][image2]
![alt text][image3]

####2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| convolution 1     	| 1x1 stride, outputs 28, 28, 6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14, 14, 6 				|
| Convolution 2  	    | 2x2 stride,  10, 10, 16 						|
| RELU          		|            									|
| Max pooling   		| 2x2 stride,  5, 5, 16 						|
| Flatten				| 400       									|
| Fully Connected		| 120       									|
| RELU  				|												|
| Fully Connected		| 84        									|
| RELU  				| 												|
|Fully Connected		|43 											|
 


####3. To train the model, I used AdamOptimizer, with 10 EPOCHs & BATCH_SIZE of 128 and 
learning rate equal to 0.001, mu = 0, sigma = 0.1


####4. I tried without normalization and 50 EPOCHS (btach size as 128) and learning rate = 0.0008, 0.0003,0.0001 without any success and then after normalization I went back to the 10 EPOCHS and 128 batch_size and 0.001 learning rate which is when I achieved more than 95% accuracy.

My final model results were:
* training set accuracy of 98%
* validation set accuracy of 98%
* test set accuracy of 97%

I chose the LeNet architechture because it best suits our needs in converging normalized images. And as the model was giving better accuracies overtime, it can be said that this model best suites our need.

###Test a Model on New Images

####1. Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
For the first (no entry ) image , as there are no other images close to this this has advantages.
For the second ( priority road) it is will be difficult if the center portion is too dull.
For third (right ahead only), can get confused with straight or right when it is unclear.
For forth (road narrow right), the center part is very confusing with the normalization method chosen and it can pick other signs with triangle as the border. 
For fith (Stop), the alphabet nature of the sign might cause difficulties as it might get mixed up with speed limits.

####2. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| right ahead 			| right ahead  									| 
| Road narrows on right | Pedestrians									|
| Stop 					| Stop											|
| priority road   		| priority road					 				|
| no entry  			|  no entry 		 							|

It predicted 4 out of 5 correctly which is 80% as opposed to the test accuracy of 97%.


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

####3. All the softmax probabilities are shown in 55th cell of Ipython notebook

| Probability         	|     Prediction	        					| Correct/Wrong | Comment |
|:---------------------:|:-------------------------------:|:------------------:|:------------------:|  
| 0.9         			| Priority Road   									| Correct| it is pretty sure here with 0.9 with all other predictions exponentially low|
| 1.0     				| No Entry 										| Correct| No Entry is seems to be very clear and nothing else is close enough to this|
| 0.8					| Pedestrians											| Wrong | This comepletely went wrong with 1st and second predictions being wrong but the last - 5th probability of 0.00009, although exp. less being correct |
| 0.9	      			| Turn right ahead					 				|   Correct| it is pretty sure here with 0.9 with all other predictions exponentially low|
| 0.9				    | Stop     							|   Correct|it is pretty sure here with 0.9 with all other predictions exponentially low|






