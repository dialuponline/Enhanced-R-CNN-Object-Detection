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

python detect.py --crop_mode=bing --pretrained_model=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel --model_def=/path/to/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt --mean_file=/path/to/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy --gpu --raw_scale=255 --weights_1st_stage_bing /path/to/BING-Objectness/doc/weights.txt --sizes_idx_bing /path/to/BING-Objectness/doc/sizes.txt --weights_2nd_stage_bing /path/to/BING-Objectness/doc/2nd_stage_weights.json --num_bbs_final 2000 --detection_threshold 0.1 /path/to/pictures