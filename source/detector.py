
#!/usr/bin/env python
"""
This a version of the caffe python detector modified by Alessandro Ferrari 
(alessandroferrari87@gmail.com).

Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows ideas in
    Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
    Rich feature hierarchies for accurate object detection and semantic
    segmentation.
    http://arxiv.org/abs/1311.2524

Bing code is available at https://github.com/alessandroferrari/BING-Objectness .
"""

import os
import cv2
import sys
import time
import caffe
import warnings
import numpy as np
from skimage import img_as_float