# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/center_2017_10_02_13_46_49_013.jpg "Center Image"
[image3]: ./examples/left_2017_10_02_13_48_47_999.jpg "Left Recovery Image"
[image4]: ./examples/center_2017_10_02_13_48_47_999.jpg "Center Image"
[image5]: ./examples/right_2017_10_02_13_48_47_999.jpg "Right Recovery Image"
[image6]: ./examples/right_2017_10_02_13_50_06_945.jpg "Normal Image"
[image7]: ./examples/right_2017_10_02_13_50_06_945_flipped.jpg "Flipped Image"
[image8]: ./examples/orig2.png "Original Image"
[image9]: ./examples/gamma2.png "Gamma Image"
[image10]: ./examples/shadow2.png "Shadow Image"
[image11]: ./examples/shadow3.png "Original Image"
[image12]: ./examples/horizon3.png "Horizon Image"
[image13]: ./history.png "Training Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission Summary

My project includes the following files:
* ```model.py``` containing the script to create and train the model
* ```drive.py``` for driving the car in autonomous mode
* ```model.h5``` containing a trained convolution neural network 
* ```writeup_report.md``` this document summarizes the results
* ```video1.mp4``` a video of the car autonomously completing a lap of track 1
* ```video2.mp4``` a video of the car autonomously completing a lap of track 2 (Bonus)

#### 2. Running the Code
Using the Udacity provided simulator and my ```drive.py``` file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

#### 3. Model Generator Code

The ```model.py``` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture

It was recommended that we examine this [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) to learn how they used images from various camera locations synthesized from a single center camera to augment a steering control network. It made sense to start with this CNN model first and see how it performed. 

My model consists of a convolution neural network with three *5x5* and two *3x3* filter sizes and depths starting at *24* and ending up at *64* (```model.py lines 142-154```).  This was then flattened to *8448* neurons and followed by *4* fully connected layers of sizes *100, 50, 10, and 1*.

The model includes RELU activation units after each convolutional layer to introduce nonlinearity, and the data is normalized to a value between *-0.5 and 0.5* in the model using a Keras lambda layer. The input layer is an image size of *160x320x3*. This was cropped *40* pixels off the top and *20* off the bottom to remove extraneous information that did not need to be learned (```model.py lines 140-141```).

#### 2. Attempts to reduce overfitting in the model

Employing the Keras model.fit function's return value, a history object was saved and used to plot a training loss diagram.  This plot showed that the validation loss increased over time, but the training loss continued to decline which is a sign of overfitting.  To combat the overfitting effect, I added dropout layers after each of the dense layers in order to successfully reduce overfitting (```model.py lines 149, 151, 153```). I had initially only added dropout after the convolution layers, but as mentioned in the model parameter section, drop was needed after the dense layers to avoid another overfit issue.

The model was trained and validated on different data sets provided by the Keras validation_split property to ensure that the model was not overfitting (```model.py line 123```). *20%* of the data was held back for the validation set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, but the default learning rate could still be modified, and it was found that the default value of *1e-3* resulted in an optimal training loss curve, which was more gradual and less boot shaped.

Sometimes the model would erroneously converge to an incorrect local minima, resulting in a constant steering value being used to drive the car (usually off the road). This was due to overfitting and was solved by adding droput after the dense layers.  I later removed the dropout after the convolution layers without any noticable effect as it was not needed anymore.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as suggested in the project guidelines. 

For details about how I created the training data, see the next section.

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first use a very simple architecture to make sure I could see at least some success with my training data and then move to a more complicated one.  

At first I used a simple single layer network which flattened and provided a single output node from the normalized input images.  I knew this network was too simple to drive the car completely around the track, but it was useful to see that it did keep the vehicle, more or less, in the middle.

In order to increase the complexity of the network I needed to have activation units that could handle a non-linear driving problem.  Using the suggested [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as a guideline, I knew it could handle the task at hand.  Rather than build a model from scratch, although academically interesting, it made sense to adapt one already proven to work in the field. So, instead of model driven approach, a more data driven approach was used to solve the problem. For details about the data driven approach see the following section 3.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contained dropout layers after each of the convolution layers as well as the dense fully connected layers.  A dropout value of *0.5* (half the activations) seemed to do the job.

The final step was to run the simulator to see how well the car was driving around track one.

The gains of using the NVIDIA model were profound.  The car actually looked like it was being driven by a human being. It demonstrated smoother motions, less twichy turning angles, and a greater sense of direction managing the turns.

Although it was now 'powered by NVIDIA' the vehicle still did not perfom as expected even on straight roads often verring off the track for no apparent reason.  I found that the problem was that the model was being trained with BGR color channels but the program used to inference the model ```drive.py``` was using RGB color channels.  Once I forced opencv to convert the images to RGB before training the car started driving much better. 

The car still has a tendancy to overshoot corners and fall off the track, or when it got close to the edge of the road it would never fully recover but continue to drive straight on top of the road edge until it hit a barrier.

To overcome the recovery senario, I created additional training data starting at the edge of the road correcting the car back to center.  To combat the overshoot, it was neccessary to remove some of the data driving down a road with no turn angle.  This prevented overfitting the 0 degree turn angle.

Later, while running on the challenge track two, it was found that the shadows on the road would sometimes cause the car to drive into the barriers.  This was fixed by injecting false shadows into the training data to help it generalize the road a bit better but this adversely affected performance on track one. I figured that this was due to the lack of shadowing on track one, so I reduced the frequency of adding shadowing to *50%* of the images and this allowed it to navigate both tracks.

Another problem I found while navigating track two was that the car failed to turn enough on a particular uphill corner which consisted of mostly blue sky behind the barriers.  This was a bit unusual and something not seen in the rest of the training set. I tackled this problem in several ways.  

First, I increased the steering constant for left and right camera views to give a better chance to turn corners, and this helped in all cases but this one I mention above. Next, I added horizon jitter to the generator in an attempt to have the model generalize more when climbing uphill and downhill, but this still did not allow it to make the corner.  Then, I moved the mask cut-off for the horizon a bit further up to allow a better view to detect the barricade flags, but this still did not fix the turn completely. 

Finally, I decided to increase the number of epochs from 20 to 30 and see what more training could accomplish. This last step worked well, and the car had learned enough to make the uphill turn with no problems. However, from training set to training set, the car could still struggle with this corner.  In the end, I decided to add more training samples for this corner, but I had wanted to solve it with the network architecture and augmentation because, in the real world word, it may have been impracitcal to head back out and collect more data due to costs, weather conditions, or time frame.

I let the model train for *100* epochs, and at the end of the process the vehicle is able to drive autonomously around both tracks without leaving the road. However, it showed a high degree of occilations left and right when trying to drive on straight roads.  I halfed the steering offset from *.25* to *0.125* and trained the network again for *100* epochs. The occilations stopped and the car had even lower training loss overall.


#### 2. Final Model Architecture

The final model architecture (```model.py lines 139-154```) consisted of a convolution neural network with the following layers and layer sizes.

|Layer (type)                     |Output Shape          |Param #     |Connected to    |
|:-------------------------------:|:--------------------:|:----------:|:--------------:|
|lambda_1 (Lambda)                |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]
|cropping2d_1 (Cropping2D)        |(None, 100, 320, 3)   |0           |lambda_1[0][0]
|convolution2d_1 (Convolution2D)  |(None, 48, 158, 24)   |1824        |cropping2d_1[0][0]
|convolution2d_2 (Convolution2D)  |(None, 22, 77, 36)    |21636       |convolution2d_1[0][0]
|convolution2d_3 (Convolution2D)  |(None, 9, 37, 48)     |43248       |convolution2d_2[0][0]
|convolution2d_4 (Convolution2D)  |(None, 7, 35, 64)     |27712       |convolution2d_3[0][0]
|convolution2d_5 (Convolution2D)  |(None, 5, 33, 64)     |36928       |convolution2d_4[0][0]
|flatten_1 (Flatten)              |(None, 10560)         |0           |convolution2d_5[0][0]
|dense_1 (Dense)                  |(None, 100)           |1056100     |flatten_1[0][0]
|dropout_1 (Dropout)              |(None, 100)           |0           |dense_1[0][0]
|dense_2 (Dense)                  |(None, 50)            |5050        |dropout_1[0][0]
|dropout_2 (Dropout)              |(None, 50)            |0           |dense_2[0][0]
|dense_3 (Dense)                  |(None, 10)            |510         |dropout_2[0][0]
|dropout_3 (Dropout)              |(None, 10)            |0           |dense_3[0][0]
|dense_4 (Dense)                  |(None, 1)             |11          |dropout_3[0][0]

```sh
Total params: 1,193,019
Trainable params: 1,193,019
Non-trainable params: 0
```

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

#### Center Camera
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

#### Left, Right, and Center Cameras

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from when it is about to leave the road. These images show what a recovery looks like starting from left, center, and right cameras:

![alt text][image3]
![alt text][image4]
![alt text][image5]

A steering angle offset was needed to correct the ground truth because these images were fed into the network as if they were coming from the center camera.

#### Flipping Camera

To augment the data sat, I also flipped images and angles thinking that this would avoid overtraining the model to a left or right turn angle bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


#### Light and Shadow

This model could not generalize to track two, so I repeated the above process on track two in order to get more data points.  This helped, but the car still had issues with some of the shadowed turns.

I added random brightness and artificial shadows to help it generalize between the sunny track one and the shadowy track two enviroment. Here is an example of an original image on the left, brightness adjustment in the center, and finally a shadow applied on the last:

![alt text][image8]
![alt text][image9]
![alt text][image10]

#### Horizon Shift

Track two introduced climbing and descending hills, as well as bumpy roads, which resulted in some bounce in the car.  In order to augment the data for these conditions a random horizon shift between *-15* and *+15* pixels was applied. An example of this is shown with the original image on the left: 

![alt text][image11]
![alt text][image12]

After the collection process, I had *19855* data points. *15884* were used for training and *3971* samples (*20%*) were set aside for validation.  I then used five times the training size (*79420* samples) to train the CNN at each epoch.  To work with such a large data set I used a Keras generator to load in and augment the training set in parallel with the training process (````model.py lines 54-99```).

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was greater than *100* as evidenced by the training history plot below. The training loss levels out after epoch *30* but, on average, does not seem to increase until about epoch *85*. The training loss still continued to fall but started leveling out, so I let it go to *100* epochs and stopped it there. I used an Adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image13]

### Track Two and Conclusion

The model was trained with images and augmentation from track one and applied to track two.  Unfortunately, the car did not perform well enough to complete the track. I decided that gathering training data from both tracks to train a model that could drive equally well on track one and track two was the best path forward.

After training the final model (model.h5) the car performed very well on both tracks consitantly. It was interesting to note that the car picked up some of my driving behaviours as well.  I had a tendency to go wide just before a turn which I saw the trained model doing as well.

This was a very fullfilling project to work on.  It is very impressive that one can train a car to drive in a simulated environment by learning from your own actions.  Working on the dataset, both augmenting and manipulating images, instead of worrying about network architecture (other than adding droput) was a good strategy.  

