#!/usr/bin/env python
"""
This a modified version of the object detector available from caffe.
A good starting point thus is to take a look at the original version:
http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb
The version has been modified by Alessandro Ferrari (alessandroferrari87@gmail.com).

detect.py is an out-of-the-box windowed detector callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

Bing code is available at https://github.com/alessandroferrari/BING-Objectness .

Example usage:

python detect.py --crop_mode=bing 
--pretrained_model=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel 
--model_def=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt 
--mean_file=/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy --gpu 
--raw_scale=255 --weights_1st_stage_bing /path/to/BING-Objectness/doc/weights.txt 
--sizes_idx_bing /path/to/BING-Objectness/doc/sizes.txt 
--weights_2nd_stage_bing /path/to/BING-Objectness/doc/2nd_stage_weights.json 
--num_bbs_final 2000 --detection_threshold 0.1 /path/to/pictures/image.jpg 
/path/to/results/output.jpg /path/to/caffe/data/ilsvrc12/det_synset_words.txt
"""
import os
import cv2
import time
import json
import caffe
import argparse
import numpy as np
from random import randint
from skimage.io import imread
from detector import Detector, resize_image

CROP_MODES = ['bing']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "image_file",
        help="Input image filename."
    )
    parser.add_argument(
        "output_image_file",
        help="Output image filename. Format depends on extension."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="bing",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--weights_1st_stage_bing_fn",
        default=None,
        help="Weights generated for the first stage of bing."
    )
    parser.add_argument(
        "--sizes_idx_bing_fn",
        default=None,
        help="Indeces of the active sizes for bing."
    )
    parser.add_argument(
        "--weights_2nd_stage_bing_fn",
        default=None,
        help="Weights generated for the second stage of bing."
    )
    parser.add_argument(
        "--num_bbs_psz_bing",
        default=130,
        type=int,
        help="Number of bounding boxes per size index in bing."
    )
    parser.add_argument(
    