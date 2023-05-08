# Enhanced-R-CNN-Object-Detection

Enhanced object detection method using simplified Python-caffe implementation of R-CNN. This implementation optimizes the bounding box proposal with Pythonized BING, making it faster and more user-friendly than the original MATLAB version.

## Getting the model

Visit http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb to get the model and related files.

## Dependencies:

- Python
- Caffe
- numpy
- opencv
- scikit-image
- BING-Objectness (available at https://github.com/alessandroferrari/BING-Objectness )

## Usage:

After moving to the repository folder on your command line, execute the following:

- cd source
- python detect.py -h

This will provide a complete synopsis of the program.

## Use case:

Here is an example of its usage where path variables should be replaced with your specific file paths:

python detect.py --crop_mode=bing --pr