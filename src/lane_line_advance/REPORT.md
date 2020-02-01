# Finding Lane Lines on the Road
--------------

Goal: This project is aimed to find lane lines and segmentation the lane.

In highways following rules of lane lines are important for maintain a safe driving. Highways however have turns and 
long view can 
  
 Below are the topics.
 * Tools to build a partly robust yet simple lane line detection algorithm that works on many cases.
 * Challenges with the simple algorithm and techniques to overcome them.
 * Combine tools to build a more robust algorithm.
 * Yet more challenges.
 
 
## Partly Robust yet simple Algorithm (Tools Required):
--------------------

#### Camera Calibration (Taking care of Distorted images):
Cameras with lenses have curvy edges that tends to distort the image on the edges. These distortion can hinder proper
 conversion of 2D image points to the 3D real world surface. There are mainly two types of distortions that the cameras are more susceptible to,

   * Radial Distortion: Radial distortion are the type of distortion that warp and curves the edges of an image.
   * Tangential Distortion: Here the image suffers from shifts and alignment, due to misalignment with the camera
   
In-order to correct for radial distortion we need to at-least learn 3 parameters (k1, k2, k3), and to correct 
for tangential distortion we need to learn two parameters (p1, p2). We learn these parameter using a set of 
Chessboard images and apply it to all the frame of the lane line video. This takes care of the distorted edges
 
#### Gradients and Color Spaces Threholding:
Color are a very good indicator of a lane line provided the lane lines are visible. Lane lines are mostly while or 
yellow is color. 

*RGB*: The Red and Green spectrum of the RGB color space does a pretty good job in finding the while lane 
lines but more often fails in identifying yellow light.

*HLS*: The S "saturation" spectrum of the HLS color space does a very good job in identifying the yellow lane line 
and does descent with the white lane lines.

*Gradients*: Gradients are the best way to find lines given the image is preprocessed pretty well. Also, gradient in the x 
direction see, to be more effective because lane lines tend to go straight We apply *SobelX* to determine gradients 
on the image preprocessed with above techniques.

*Approach*: S channel picks up the lane but also has relateively high value for nearby lane objects such as cars or 
far trees. The Red channel or the L channel is pretty good at filtering them. We use a simple thresholding heuristic 
on both R and S channels to get most out of the visible lane lines and use gradients on LS channels to get rid of 
some noise.

![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/preprocessed_image.png)

#### Perspective Transform (Warping the image for better prediction):
Lane lines in real world are parallel, but they tends to meet as we go further in the image. Fitting a polynomial 
curve to lane lines in this case can would not result in a very good estimate, additionally it would be more prone to
 errors and noise in the image. Changing the perspective to birds-eye-view would make the lane lines approximately 
 parallel and easier to fir a polynomial.
 
 Perspective transform is a technique to transform the image perspective usin image points . 
 
 Step 1: Take an approximate view (trapizium) that contains both the lane lines.
 Step 2: Use prespective transform to change the view (trapizium) to the birds-eye perspective
 
![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/perspective_transform.png)
 
#### Estimating lane line region to fit a better polynomial:
Gradient are noisy when taking into account the surrounding. It is important to find an approximate region where we 
are more certain of find the lane. Despite the noise, we can assume that the gradients would be more accumulated in 
the lane line region. So how do we find them? 

##### *Histogram Distribution (Finding where both the lane originate from)*
Assuming lane lines are approximately 
parallel to 
y-axis, we can sum the values along y-axis and plot a histogram. We then divide the histogram into two half and take 
the x-value of the peak points in both the half. Say image_size = 720x720, half_1=720x460 and half_2=720x360 and peak
 points for half_1=(720,160) and half_2=(720, 540). Then we consider that our left lane originates from (720,160) and
  right lane originates from (720, 540).
  
![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/histogram_dist.png)

##### *Finding Lane Region*
We use a sliding window technique. Here we fit a window at the origin and slide it through the y-axis where the gradient are accumulated more.

**Fitting a Polynomial** 
At this point we have the lane points i.e the points under the windows. Under the assumption that our sliding window 
technique is a good approximation, we take all the points and fit a 2nd order polynomial. Voila! we found the lane 
lines.

Now we simply take a buffer around the fitted polynomial and consider it our lane line.

![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/curvature_windows.png)


#### Unwarp the image and iterate:
Now that we have our lane line is warped image, we convert them back to the actual image space by using perspective 
transform and plot the segmentation mask on the estimated lane boundary.

*Iterate* We can iterate this whole process over and over again for all the frames, however we can skip the 
step **Estimating lane line region to fit a better polynomial**. Because lane line do not change a lot for several 
consecutive frames. Therefore, after finding the polynomial in the first frame, we can simply use a buffer region 
around the polynomial and fit the new polynomial using the points in that buffer region.

![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/postprocess.png)


## A more Robust way (Minor improvements on the above techniques)
------------- 
  
#### Finding Histogram Distribution:
For many cases just summing the y axis values of the binary-warped preprocessed image may return good estimation of 
the location where we should start the sliding window technique, but this is not always true (In challanging 
scenarios where the bottom of the image has high gradients across the entire x axis this approach would fail). Here 
we employ a simple weighting criteria. We have a prior knowledge of roughly where the lanes should originate. 
Therefore, here we create a weight matrix based on our prior knowledge and multiply it to our binary-warped 
preprocessed image. After, that we take the sum across y axis. 

Below is an image of such matrix, THey yellow the region the more weight that region has  
    
![Preprocessing-Img](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_line_advance/image/histogram_weights.png)
   
#### Lane Smoothing (Moving average) and lane swapping:

*Lane Smoothing*: THe lane can fluctuate from frame to frame. But in the real-world case this may not be true. The 
fluctuation of lane lines are primarily because of bad gradients or bad pre-processing. In this case we apply a 
simple heuristic to average or perform weighted average of n-t consequtive frames. 

*Lane Swaping*: Sometimes shadows and excessive lighting can prohibit the proprocessing algorithm to find lanes or 
rather find very few points (which can diverge a lot from the actual lane line). In 
such a case we simply return the lane from the previous frame. If the patter appears repeatedly for many frames, we 
perform dynamic preprocessing 
      