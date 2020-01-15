# Finding Lane Lines on the Road

Goal: This project is aimed to find Lane Lines on the road.

Finding lane line is the very basic tool needed for self-driving car. This project takes into consideration, the most
 basic computer vision techniques to find lanes lines. 
 
 Below are the topics.
 * Ideas to build a simple Lane Line detection tool.
 * Challenges with the current pipeline.
 * Can we do better? Discuss some ideas.
 
 #### Ideas to build simple Lane Line detection tool:
 -----------------
 As every machine learning project we would discuss the following *preprocesing*, *model_fit*, *postprocessing*, 
 *prediction* 
 
 * **Preprocessing**:
    Lane lines are ideally preprocessing would include cleaning the image   
    1. Convert the RGB image to gray scale.
    2. Use Canny-Edge detection to find all the edges in the image. Lines tend to have sharp edges.
    
    ![Self-Driving Car Simulator](https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/images/preprocessing.png)
    
    3. Masking: The left and right though are parallel in the real world, tend to converge are the reach the center of 
    the image. Here we take a simple heuristic to mask the image where there is high probability of finding only the 
    lanes.
    4. Hough Lines: Lane lines tend to be straight provided the Road is straight. So we use Hough transform lines to
    find the lanes.
    
    <div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/image/hough_lines.png" width="300" height="300"><figcaption><center>Hough Lines</center></figcaption>
      	    </td>
        </tr>
    </table>
    </div>
    
    Challange: We can see that the hough lines are pretty good in finding lines but fail to extend it to the entire 
    view point. Having a full straight line is important for an algorithm to drive safely   
    
 * **Model Fit**:
 A line is simply the equation **y = mx + c** were **m is slope** and **c is the intercept**. The end goal is to come
  up with a good slope **m** and intercept **b** such that we can fit the line equation and find our left and right 
  lane. As of hough 
  lines
  we 
 have multiple **y** and **x** points, and so we can find **m** and **c**.
    1. *Gather lines for each lane (left nad Right)*: Here we use a simple heuristic and say that line to the left of
     the center belongs to left lane and line to the right of image center belongs to right lane.
    2. *Filter lines for each lane*: Lines in left lanes would most likely have positive slope and right lanes are 
    likely to have negative slope. Here we reject all hough lines that doesn't meet the slope criteria. Moreover, in 
    order to find robust slopes, we need to get rid of lines that have slope which can be considered as outlier. So 
    we simple add a low and high threshold for slope and reject every line that doesn't meet the threshold.
    3. *Aggregative Lines and Slopes*: To fit a line, all we need is a point (x, y) and the slope (m). Then we 
    calculate the intercept and construct the line by extrapolating. A simple way to find the slope **m** and the 
    points **(x, y)** is to take the mean of all the slopes and all the points. However, outlying points can affect the 
    mean value, hence it is a good idea to take the median value.
    4. *Fit* Once we have out slope and point in the line we simply calculate the intercept and generate **x** values
     for every **y** value in the field of view.
    5. *PolyFit* A more simple way to extrapolate points and fit a line is to use the **numpy.polyfit()** function, 
    which takes input points **(x, y)** that can be gathered from Step 2.
    
    <div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/self-driving-vehicle/blob/master/src/lane_lines/images/lane_line_extrapolated.png" width="300" height="300"><figcaption><center>Liner Extrapolation of Hough Lines</center></figcaption>
      	    </td>
        </tr>
    </table>
    </div>
    
 * **Postprocessing**   
    All the heavy lifting is already done in the **Preprocessing** and **Model Fit** stage. Postprocessing 
    simply includes steps that take care of rendering the output is a presentable way and taking care of some edge 
    cases.
    1. A line by itself has a width equivalent to 1 pxl which is bad to have in edge cases. Here, we thicken the line  
    by adding a buffer on both the side.
    2. While running the algorithm for consecutive scene, there can be cases that the preprocessing step didnt output
     reliable segments and all the lines were filtered in the process. In such a scenario we simply use the 
     parameters from the previous frame, under the consideration that the change in two consecutive frame is minimal.    
    
    
#### Challenges with the current pipeline.
--------------
The current pipeline using a very basic algorithm of line fit and only works for a particular use case. Below are 
some of the use cases that the current pipeline fails to deliver a robust outcome.
 
   * *Curved lines*: Since we fit a straight line equation our algorithm wold fail for sharp turns in the road. An easy
    to tackle the problem would be to fit a curve line using 2nd order polynomials. 
   * *Shadows and lighting* Edge detection suffers a lot under varying lighting conditions. In such a scenario one 
   could explore edge detections on different color spaces such as the HLS (Hue, Lighting and Saturation)
   
   
#### Can we do better.
-------------
There are many tools in computer vision that can be leveraged and put together to make a robust algorithms. Some
 of them are discussed below.

   * Gradient Magnitude: While Edge detection is a general case that uses gradients in x nad y direction, it would be a 
   better idea to use gradient in x direction since lane line runs parallel to the y axis.
   * Gradient Orientation: We can also introduce prior on gradient orientation since lane lines in most cases would 
   be between 45 - 135 degrees.
   * Color Spaces: Introducing different color spaces can help with varying light condition.
   *  
 
  