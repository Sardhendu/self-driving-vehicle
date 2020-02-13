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
    - ROI Curve
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
 
![Class-Distribution-Plot](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/traffic_sign_classifier/images/class_distribution_plot.png)


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
            - Epoch = 35
            - Train Steps = 34799/256  (total_training_data/batch_size)
   2. **Model Pipeline**:
        - *Optimizer*: Since we use a relatively large batch size of 256 a good optimizer to use would be Adagrad, 
        however in our case we choose Adam Optimizer since Adam is said to work bet in many scenarios. 
        - *Learning Rate Scheduler*: We use a variation of cosine annealing and polynomial decay combined. The idea is 
        to bump up the learning rate in the 1st few thousand steps so that model learns the most from the dataset and
         then decay using cosine annealing. Below is a plot of the learning rate decay function  
   3. 