

1. Random Brightness and Contrast would take care of edge cases when the image is not clear, say foggy weather
2. Random Hue may not be the best idea because colors are important features of these signals, and messing with the 
colors spaces could cause other problems. Messing with random hue can represent a red color as green and we know 
the importance of red vs green in a traffic signal. However, adding only a little bit of randomness wont hurt
    - Say we add a reasonable 0.05 randomness to red. This may take the value to orange or Magenta. Depending of 
    the color wheel.
3. Saturation: Saturation changes the intensity of the color, which can benefit out model. Say we dont have examples 
of newly installed stop signs were the intensity of red color is high. Adding saturation to a dull red can make it new.