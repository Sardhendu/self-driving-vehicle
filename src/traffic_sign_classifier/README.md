# Project: Traffic Sign Classifier

This project implements the Le-Net architecture to classify Traffic Signs.

  * Dataset
  * Image Pre-processing - Augmentation
  * Model Architecture 
  * Train/Eval Process
    - Dataset (input_fn)
    - Learning Rate Scheduler
    - Optimizer
  * Eval/Test Metric
    - Accuracy
    - Confidence Matrix
    - Top-5 Prediction output
  * Challenges and Improvements

### Final Metric

## Install and Run:
1. **Clone Repo**: git clone [git@github.com:Sardhendu/self-driving-vehicle.git]()
2. Install [*Pipenv*](https://pipenv-fork.readthedocs.io/en/latest/)
3. cd self-driving-vehicle
3. **Install Libraries**: pipenv sync
4. Download **test_images**, **test_videos**, **test_videos_output** from [Udacity's Repo](https://github.com/udacity/CarND-LaneLines-P1) and place it inside  **/self-driving-vehicle/src/lane_lines/data/** 
5. A simple way to start is to run the Jupyter Notebook [P1.ipynb](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/P1.ipynb)
6. Abstracted Methods can be found at [tools.py](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/tools.py)

### Dataset
The dataset belong to the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The data set contains more that 50,000 labeled images and can be [downloaded](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip) here. The dataset contain 43 different traffic signals and is prone to class imbalance. 

Data size:

   - Train Size: 34799
   - Validation Size: 4410
   - Test Size: 12630
   
Class Distribution
 
![Class-Distribution-Plot](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/class_distribution.png)


## Preprocessng/Image-Augmenation:

Note: All the images below can be found in a pair. The left image indicates the actual image from the dataset and 
the right image indicates the preprocessed version.

1. ***Brightness and Contrast*** would take care of edge cases when the test image is not clear, say foggy weather,
 or darker image. etc
 
![Random-Brightness](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/random_brightness.png)
 
![Random-Contrast](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/random_contrast.png)
  
2. ***Random Hue*** may not be the best idea because colors are important features of these signals, and messing with 
the colors spaces could cause other problems. Messing with random hue can represent a red color as green and we know 
the importance of red vs green in a traffic signal. However, adding only a little bit of randomness wont hurt
    - Say we add a reasonable 0.05 randomness to red. This may take the value to orange or Magenta. Depending of 
    the color wheel.
    
3. ***Saturation***: Saturation changes the intensity of the color, which can benefit out model. Say we dont have 
examples of newly installed stop signs were the intensity of red color is high. However in out training data we 
have many stop sign with low intensity color pxls. Adding saturation to these image can generalize well in many 
unknown cases. 
 
![Random-Saturation](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/random_saturation.png)

4. ***Zoom***: Zoom is a good feature to handle taffic sign at different scales. 
 
![Random-Zoom](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/random_zoom.png)

5. ***Affine Transform***: Affine Transform is governed by three transformation 
   * 1) *Translation (tx, ty)*: Governs the horizontal and vertical shifts
   * 2) *Rotation (angle)*: How much the image rotates along the axis
   * 3) *Scale (s)*: Increase/Shrink the image size
   
   Since we develop a custom affine transform function (Tensorflow currently doesn't have one and keras ops are not 
   supported inside tensorflow dataset pipeline), the output of the image may be a little weird. Cutting to the 
   chase, we create a homogeous transform matrix using linear combination of each transform matrix and multiply it to
    all our all out pixels values in the image.
    
![Random-WarpAffine](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/warp_affine.png)

6. ***Random Shift*** Random shift is simply the Translation (tx, ty). 

![Random-Shift](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/random_shift.png)

7. ***Scale***: We simply divide all the pxl values by 255.

## Model Architecture: Le-Net:
Le-Net is a simple convolutional neural network architecture with two convolutional layers, each followed by 
relu activation and max-pooling layers. The architecture also contains two Fully connected layer at the end. Below is a 
simple flow.

    Conv((5,5), 6) -> Relu -> Pool(2,2) -> Conv((5,5), 16) -> Relu-> Pool(2,2) -> Fc1(120) -> Fc2(84)
    
We use Xavier initialization to initialize all the network weights.
    
## Train/Eval Process:
The Training Process is very modular and broken into 5 major parts.

   1. **Dataset Pipeline**:
        - Here we create a tensorflow dataset pipeline that parses records, performs preprocessing on images and 
        outputs the feature and labels to the model
            - Batch Size = 256
            - Epoch = 300
            - Train Steps = 34799/256  (total_training_data/batch_size)
   2. **Model Pipeline**:
        - *Optimizer*: Since we use a relatively large batch size of 256 a good optimizer to use would be Adagrad, 
        however in our case we choose Adam Optimizer since Adam is said to work bet in many scenarios.
        
        ![Loss](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/loss.png) 
        
        - *Learning Rate Scheduler*: We test with polynomial decay and with a variation of cosine annealing and 
        polynomial decay combined. The idea is to bump up the learning rate in the 1st few thousand steps so that model learns the most from the dataset and
         then decay using cosine annealing. Both the decay gives relative performance, for our final model we choose 
         polynomial decay. Below is the plot showcasing that.
        
        ![Learningrate-Schedular](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/learning_rate.png)
         
         - *Weight Decay*: We use weight decay to control overfiting. Additionally, augmentation in the dataset also 
         prevents the model to overfit.

## Eval/Test Metric:
   * Eval Accuracy: 96.6%
   * Test Accuracy: 95.2%
   * We observe that we have a high class imbalance, hence accuracy may not be a good measure of metric, rather we 
   can use precision recall. Below are the plots for accuracy, precision and recall
   
   * Validation Metric:
![Accuracy](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/accuracy.png)

![Precision-Per-Class](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/precision.png)

![Recall-Per-Class](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/recall.png)

   * Test Confidence Matrix: 
![Confidence-Matrix](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/confidence-matrix.png)


## New Test Images:

![New-Unseen-Images](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/test_images.png)


1. ***Right Turn***: This image is self explanatory, the image says right turn and the model says  
```
    Running for path: ...... ./predict_images/right_turn.jpeg
    Pred Labels:  [33 37 39]
    Pred Classes  ['Turn right ahead', 'Go straight or left', 'Keep left']
    Pred Scores:  [1. 0. 0.]
```
  
2. ***Right Turn with yield*** I like this one because this image contain two signs. The model surprisingly output 
**"Dangerous curve to the right"**. This to me makes intuitive sense because German Traffic Sign for Dangerous curve 
to the right would have a **"right arrow""** drawn inside the **yield** sign. Since the model probability for the 
sign is low we can 
feel safe in assuming that we didn't have similar image in our training data. Now, we know convolutional network are 
not very good at localizing one object wrt other objects, they strive to learn features, moreover we use LeNet which 
is one of the smallest net, so we can say that the model didn't have enough power to learn the global context. Using 
this information one
 can assume that the model learned two distinct feature 1) right turn 2) yeild. Now it combined the information 
 independently to output **Dangerous curve to the right** which kinda has both the feature.   
 
```
    Running for path: ...... ./predict_images/right_turn_yield.jpeg
    Pred Labels:  [20  2 23]
    Pred Classes  ['Dangerous curve to the right', 'Speed limit (50km/h)', 'Slippery road']
    Pred Scores:  [0.52 0.29 0.08]
```
 
3. ***Padestrian (Childern's crossing)*** Well I downloaded this images thinking of padestrians. After teh model 
output convinced me more. The output makes sense given the image

```
    Running for path: ...... ./predict_images/padestrian_2.jpeg
    Pred Labels:  [28 11 23]
    Pred Classes  ['Children crossing', 'Right-of-way at the next intersection', 'Slippery road']
    Pred Scores:  [0.95 0.05 0.  ]
```
 
4. ***Speed Limit 120***: Now this one is easy. The model does pretty good at finding the right class. However, the 
model didn't assign a very high probability to the correct class. Reason being, like we discussed in **Right Turn 
with yield**, these shallow networks are not very good at determining relationship between individual features rather
 treat them independent. That's the reason we see, **"Speed limit 100 (for 1 and 0 in 120)"** and **"Speed limit 20 
 (for 2 and 0 in 120)"**. Now you may ask why higher probability to 100 not 20, as simple way to think is that 20 has
  only **two** features (2 and 0) whereas, 100 has **three** features.
    
```    
    Running for path: ...... ./predict_images/speed_limit_120.jpeg
    Pred Labels:  [8 7 2]
    Pred Classes  ['Speed limit (120km/h)', 'Speed limit (100km/h)', 'Speed limit (50km/h)']
    Pred Scores:  [0.66 0.34 0.  ]
```

5. ***Stop Sign*** The model does say this as stop sign. However, if I don't crop this image to the boundary, but 
include some of the background the model gets confused as says 'Speed limit (60/h)' as the output. Which makes sense 
because including background and resizing the image to 32x32 would take away the resolution and the model may think 
that its a **STOP** is **60** taking **S and O** as its highest weighted feature. Well this doesn't sound very 
convincing but given the fact the our dataset is **imbalanced** and biased towards **speed limits** this scenario 
tends to indicate that the model is biased in some sense. So what now, do we have a bad model, well one way to go 
would be to upsample minor classes, so that the model can see them more times.
 
```
    Running for path: ...... ./predict_images/stop.jpeg
    Pred Labels:  [14  1  0]
    Pred Classes  ['Stop', 'Speed limit (30km/h)', 'Speed limit (20km/h)']
    Pred Scores:  [0.51 0.49 0.  ]
```

References:

  * Dataset-Paper : https://www.sciencedirect.com/science/article/abs/pii/S0893608012000457
  * 