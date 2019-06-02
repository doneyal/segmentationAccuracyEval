
# Evaluate Instance Segmentation
Calculate Accuracy of a segmentation output (like Mask RCNN) based on an annotated reference

# How to run
"python segmentationAccuracyEval.py -l sample/legend.json -a sample/annotated_image.png -i sample/input_image.png -m sample/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28 -op output.json"

model : Any  segmentation model given in [Tensorflow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) can be used 

