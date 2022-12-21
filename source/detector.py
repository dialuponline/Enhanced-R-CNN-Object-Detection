
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
try:
    from bing import Bing
    bing_flag = True
except Exception:
    warnings.warn("Impossible to import bing.")
    bing_flag = False

def resize_image(image, reference_edge = 512.0):
    
    if image.ndim==3:
        h,w,nch = image.shape
    else:
        h,w = image.shape
    max_edge = max(w,h)
    ratio = float(reference_edge) / max_edge
    new_w = ratio * w
    new_h = ratio * h
    image = cv2.resize(image,(int(new_w),int(new_h)), interpolation=cv2.INTER_CUBIC)
    
    return image

class Detector(caffe.Net):
    """
    Detector extends Net for windowed detection by a list of crops or
    selective search proposals.
    """
    def __init__(self, model_file, pretrained_file, gpu=False, mean=None,
                 input_scale=None, raw_scale=None, channel_swap=None,
                 context_pad=None, weights_1st_stage_bing = None, sizes_idx_bing = None,
                 weights_2nd_stage_bing = None, 
                 num_bbs_psz_bing = 130, num_bbs_final_bing = 1500):
        """
        Take