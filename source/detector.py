
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
        gpu, mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        context_pad: amount of surrounding context to take s.t. a `context_pad`
            sized border of pixels in the network input image is context, as in
            R-CNN feature extraction.
        """
        caffe.Net.__init__(self, model_file, pretrained_file)
        self.set_phase_test()

        if gpu:
            self.set_mode_gpu()
        else:
            self.set_mode_cpu()

        if mean is not None:
            self.set_mean(self.inputs[0], mean)
        if input_scale is not None:
            self.set_input_scale(self.inputs[0], input_scale)
        if raw_scale is not None:
            self.set_raw_scale(self.inputs[0], raw_scale)
        if channel_swap is not None:
            self.set_channel_swap(self.inputs[0], channel_swap)

        self.configure_crop(context_pad)
        
        if bing_flag and not weights_1st_stage_bing is None and not sizes_idx_bing is None and not weights_2nd_stage_bing is None:
            self.bing = Bing(weights_1st_stage = weights_1st_stage_bing, sizes_idx = sizes_idx_bing ,weights_2nd_stage = weights_2nd_stage_bing, num_bbs_per_size_1st_stage= num_bbs_psz_bing, num_bbs_final = num_bbs_final_bing)
        else:
            self.bing = None
        
    def detect_bing(self, image):

        assert not self.bing is None
        
        if not bing_flag:
            print "Bing detection invoked but error while importing bing module!"
            sys.exit(1)
                
        
        t0 = time.time()
        bbs, scores = self.bing.predict(image)
        t1 = time.time()
        print "Bing prediction: {0:.2f}s.".format(t1-t0)
        images_windows = self.detect_windows(image, bbs)
        
        return self.get_predictions_from_cropped_images(images_windows)
        
    def detect_windows(self, image, bbs):
        """
        Do windowed detection over given images and windows. Windows are
        extracted then warped to the input dimensions of the net.

        Take
        images_windows: (image filename, window list) iterable.
        context_crop: size of context border to crop in pixels.

        Give
        detections: list of {filename: image filename, window: crop coordinates,
            predictions: prediction vector} dicts.
        """
        images_windows = []
        