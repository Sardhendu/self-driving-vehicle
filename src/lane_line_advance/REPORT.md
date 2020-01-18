

Camera Calibration:

Cameras with lenses have curves on the edges that tends to bend the light with project the 3d objects into the image 
place. This could many-a-times distort the image on the edges. There are two types of distortions that the cameras 
are more susceptible to,

   * Radial Distortion: Radial distortion are the type of distortion that warp and curves the edges of an image.
   * Tangential Distortion: Here the image suffers from shifts and alignment.
   
In-order to correct for these radial distortion we need to at-least learn 3 parameters (k1, k2, k3), and to correct 
for tangential distortion we need to learn two parameters (p1, p2).
   
   
   
      