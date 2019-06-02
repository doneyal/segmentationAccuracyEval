
# Evaluate Instance Segmentation
Calculate Accuracy of a segmentation output (like Mask RCNN) based on an annotated reference.

Python Packages required : numpy, argparse, time, cv2(OpenCv 3+), os, tensorflow(tested on tensorflow 1.12), json.

# How to run
"python segmentationAccuracyEval.py -l sample/legend.json -a sample/annotated_image.png -i sample/input_image.png -m sample/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 -op output.json".

model : Any  segmentation model given in [Tensorflow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) can be used.
The model file (with name "frozen_inference_graph.pb") and object classes labels list (with name "object_detection_classes_coco.txt" is expected in the model directory.

# Matching Algorithm:
The metric used is  Intersection/Union (IOU). The best match is based on maximum IOU values.

For example : If Mask RCNN output had 4 instances of class 'Car' (Car1, Car2, Car3, Car4) and annotated image had 5 instances of class 'Car' Car-A, Car-B, Car-D, Car-D, Car-E).

IOU values of all combination are calculated ie Car1 vs (Car-A, Car-B, Car-D, Car-D, Car-E), Car2 vs (Car-A, Car-B, Car-D, Car-D, Car-E) forming a 4X5 matrix, where Row1 represents IOU for Car-1 with Car-A, Car-B etc.

For each Row r the maximum value is calculated (say at r,c). If that value is the maximum for the entire column c then it is locked as a match and IOU value is saved otherwise the detection is rejected.

