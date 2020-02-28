# Behaviour Cloning:


### Output Video sneak peek
 
![](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/behavioural_cloning/image/sneak_peak.gif)


1. Miniconda Installation:
```bash

# Get Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod u+x ./Miniconda3-latest-MacOSX-x86_64.sh
./Miniconda3-latest-MacOSX-x86_64.sh

# Create environment using Conda
conda create -n behavioural_cloning python=3.7
```

```bash
# Installing eventlet cause some problems with conda
# First we gather all the channels available in our conda version

conda config --add channels conda-forge
conda install eventlet

```


## Data Collection:
Here we collect two different versions of data.
1. Good Data: Here we collect data while we drive the car properly on the center of the road
2. Mixed Data: Here we capture the edge cases by letting the car go off to the edge of the road and 
bringing it back to the center. The idea here is to let the model understand that when the car is driven away from the center the
 car it to be brought back to the center. 
 
## Data Augmentation and Preparation:
 * We perform horizontal flip of the images. This is essential so the model doesnt overfit by steering left, when the
  track was anticlock.
 * We make use of all cameras (left, right and center). We add a constant value +0.2 to the steering value for every 
 image with left camera and -0.2 to every image with right camera view. By adding and subtracting this small value we
  make the left and right camera behave as the center camera. This help us augment our data three times.
 * while training:
    1. We Crop the 50 pixels and bottom 10 pixels of the image, that contains trees and the sky.
    2. We normalize the data in the range of (-0.5, 0.5) 
    
    **Data Count**
    
    1. train_images=22901, train_steering_vals=22901
    2. eval_images=3053, eval_steering_vals=3053
    3. test_images=4580, test_steering_vals=4580
    
![Orig-Preprocess-Images](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/behavioural_cloning/image/input_img.png)


## Model Architecture:
We use the off-the shelf **"Xception Net"** backbone provide in the Keras Applcation Api. Though a deeper net, the use
 of Depth wise Separable convolution layers makes xception net an approachable algorithm to be run on real time. The 
 Author's of the Xception net uses two novel operation to reduce the number of parameters in the entire model, so 
 that the model can train faster with better outcomes.
 .   
 
  * **PointWise Convolution**: A 1x1 convolution is performed on he input feature map. 1x1 convolution introduce 
  significantly less parameters, while decreasing/increasing the number of channels (depth dimension).
  * **DepthWise Convolution**: Here instead of performing a fully connected convolution operation, convolution 
  operation is performed in each channel separately and are later concatenated.  
  
  
Link(https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3
-image-dc967dd42568)

We remove the top layer (classification layer) of the xception net and add a Dense Layer followed by a tanh activation.
    
  * **Tanh** Tanh is a suitable activation, because tanh squashes the values between -1 and 1, this serves our 
  purpose since our target steering value ranges from -1 to 1, where negative values indicate steering 
  to the left and positive values indicate steering to the right.

    * Xception -> Dense -> Tanh
    
    
## Model Training and Metric:
We train the model with below parameters/hyperparameters:

   * **Epochs**: 20
   * **Batch Size**: 32
   * **Learning Decay**: Polynomial Decay
   * **Pretrained weights**: ImageNet
   * **Loss**: Mean Squared Error (MSE)
   * **Optimizer**: Adam
   * **Metric** We use MSE in the validation set to check models performance. A monotonicaly decreasing loss is a better indication of model learning.
   
![Train-Eval-Metric](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/behavioural_cloning/image/metric.png)

## Future Works (TODO's)

  * Can we bring-in reinforcement learning into picture, but how do we get the rewards?
  *   
   

    
