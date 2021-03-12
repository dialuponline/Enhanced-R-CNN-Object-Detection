#!/usr/bin/env python
"""
This a modified version of the object detector available from caffe.
A good starting point thus is to take a look at the original version:
http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb
The version has been modified by Alessandro Ferrari (alessandroferrari87@gmail.com).

detect.py is an out-of-the-box windowed detector callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to imp