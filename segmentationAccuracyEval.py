import numpy as np
import argparse
import time
import cv2
import os
import tensorflow as tf
import json
import random


#loading mask rcnn graph
def loadGraph(graphName):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graphName, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
            return detection_graph


#intialising the graph with tensorflow session for run
def initMaskRcnn(detection_graph):
    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
    # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        detection_mask = detection_graph.get_tensor_by_name('detection_masks:0')
        return [sess, image_tensor,[detection_boxes, detection_scores, detection_classes, num_detections,detection_mask]]


def runGraph(graphContext, image):
    image_np_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, numd,masks) = graphContext[0].run(
          graphContext[2],
          feed_dict={graphContext[1]: image_np_expanded})
    return (boxes, scores, classes, numd,masks) 


def hex2BGR(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     b = int(hex[2*hlen/3:2*hlen/3+hlen/3], 16)
     g = int(hex[1*hlen/3:1*hlen/3+hlen/3], 16)
     r = int(hex[0:hlen/3], 16)
     return (b,g,r)


#read the legend file
#input - legend.json
#ouput1 - list of segmentData where segmentData=[ ObjectClass, number(instance),colorCode]
#ouput2 - list of all unique classes detected
def readLegend(fileName):
    detectionList=[]
    with open(fileName, 'r') as f:
        legendData = json.load(f)

    for colorCode,detections in legendData['legend'].iteritems():
        detection = detections.split("#")
        detection[0] = detection[0].lstrip().rstrip()
        if(len(detection) == 1):
            detection.append('0')
        detection.append(colorCode)
        detectionList.append(detection)
    #Sorting for search optimisation
    sortedDetectionList = sorted(detectionList, key=lambda x: x[0])
    itemList=[]
    currentItem="null"

    for items in sortedDetectionList:
        if(items[0] != currentItem):
            itemList.append(items[0])
            currentItem = items[0]

    return sortedDetectionList, itemList


#creates  masksfor each instace segmentation based on legend data
#input  - (legend_data, inpit_image)
#output - a list of mask where mask is a binary image(has vals 0 for background and 255 for foreground) corresedponding to each instance segmentation.

def loadAnnoData(legendData,annotatedImage):
    annoData =[]
    for dtect in legendData:
        bgr = hex2BGR(dtect[2])
        bgr = np.array(bgr, dtype = "uint8")
        mask = cv2.inRange(annotatedImage, bgr, bgr)
        annoData.append(mask)
    return annoData



#creates masksfor each instace segmentation based on ouput of mask rcnn
#input - detections from mask rcnn, input image, threasholds and labels for classes
#output1 - a list of detectedData where detectedData = [binary image(has vals 0 for background and 255 for foreground) corresedponding to each instance segmentation, class]
#ouput2 - list of all unique classes detected
def loadDetectionData(detectionData,ipImage,thresholds, labels):

    detData =[]
    (boxes, scores, classes, numd,masks) = detectionData
    detThreshold = thresholds[0]
    segThreshold = thresholds[1]

    height, width, channels = ipImage.shape
    imXMask = np.zeros([height,width], dtype = "uint8")

    for i in range(int(numd[0])):
        classID = int(classes[0][i]) - 1
        confidence = scores[0][i]
        # detections with very low confidence is rejected
        if(confidence > detThreshold):
            imX = imXMask.copy()
            box = boxes[0][i]
            x1 = box[1]*width
            x1 = int(x1)
            y1 = box[0]*height
            y1 = int(y1)
            x2 = box[3]*width
            x2 = int(x2)
            y2 = box[2]*height
            y2 = int(y2)

            imX = np.zeros([height,width], dtype = "uint8")
            mask = masks[0][i]
            #mask is created
            mask = cv2.resize(mask, ((x2-x1), (y2-y1)),interpolation=cv2.INTER_NEAREST)
            mask = (mask > segThreshold)
            color = 255

            imX[y1:y2, x1:x2][mask] = color
            detData.append([imX,labels[classID]])

    sortedetDataList = sorted(detData, key=lambda x: x[1])
    itemList=[]
    currentItem="null"

    for items in sortedetDataList:
        if(items[1] != currentItem):
            itemList.append(items[1])
            currentItem = items[1]

    return sortedetDataList, itemList


#given two masks of same size intersection/union  of foregrouds is calculated
#input - mask1, mask2
#output - intersection / union 
def findIou(refMask, detMask):

        diff = cv2.compare(refMask,detMask,cv2.CMP_NE)
        detCopy = detMask.copy()
        detCopy[np.where(detCopy == [0])] = [128]

        intersection = cv2.compare(refMask,detCopy,cv2.CMP_EQ)

        numInterSection = cv2.countNonZero(intersection) 
        numDiff = cv2.countNonZero(diff)

        iou = float(numInterSection)/float(numInterSection+numDiff)
        return iou


#function to write the output 
def writeOutput(legendData,opfileName):
    with open(opfileName, 'w') as f:
        f.write('{\n')
        for data in legendData:
            f.write("  \""+data[0])
            if(data[1] != '0'):
                f.write(" #"+str(data[1]))
            f.write("\" :")
            if(len(data)<4):
                f.write("0,\n")
            else:
                f.write(str(data[3][1])+",\n")
        f.write('\n}')

#function for visulaizing the overlap
def visualiseBlended(mask1, mask2,iouVal):
    height, width = mask1.shape
    imX1 = np.zeros([height,width,3], dtype = "uint8")
    imX2 = np.zeros([height,width,3], dtype = "uint8")
    imX1[np.where(mask1 == [255])] = [0,255,0]
    imX2[np.where(mask2 == [255])] = [0,0,255]
    cv2.imshow("detected",imX1)
    cv2.imshow("annotated",imX2)
    blended = imX1+imX2
    cv2.imshow("OverLap",blended)
    cv2.waitKey(5000)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--legend", required=True,
     help="legend file")

    ap.add_argument("-a", "--annotated_image", required=True,
     help="path to the annotated image file")

    ap.add_argument("-i", "--input_image", required=True,
     help="path to the input image file")

    ap.add_argument("-m", "--model_path", required=True,
     help="path to the input model file")

    ap.add_argument("-st", "--segmentation_thresh", type=float, default=0.3,
     help="minimum threshold for pixel-wise mask segmentation")

    ap.add_argument("-dt", "--detection_thresh", type=float, default=0.7,
     help="minimum threshold for detections")
    
    ap.add_argument("-v", "--enable_visualizer", type=bool, default=True,
     help=" enable / disable image visualizer")

    ap.add_argument("-op", "--output_file", default='output.json',
     help="name of the output file into which output is to be written")


    args = vars(ap.parse_args())
    enableVisualizer = args["enable_visualizer"]
    thresholds = [args["detection_thresh"],args["segmentation_thresh"]]
    outputFileName = args["output_file"]
    
    labelsPath = os.path.sep.join([args["model_path"],
    "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    annotatedImage = cv2.imread(args["annotated_image"])
    if(enableVisualizer):
        cv2.imshow(" annotated-image ",annotatedImage)

    image = cv2.imread(args["input_image"])
    if(enableVisualizer):
        cv2.imshow(" input-image",image)
        cv2.waitKey(10)


    legendData,itemList = readLegend(args["legend"])
    annoData = loadAnnoData(legendData,annotatedImage)


    modelPath = os.path.sep.join([args["model_path"],"frozen_inference_graph.pb"])

    graph = loadGraph(modelPath)
    modelContext = initMaskRcnn(graph)

    graphOP = runGraph(modelContext,image)
    detData,detItems = loadDetectionData(graphOP, image,thresholds,LABELS)


    #finding overlapa with annotated data instances and mask rcnn instances based on IOU and selecting the best match
    for item in detItems:
        #selecting all instances of a particular class
        detIds = filter(lambda i: detData[i][1].lower()==item.lower(), range(len(detData)))
        annoIds =filter(lambda i: legendData[i][0].lower()==item.lower(), range(len(legendData)))
        nmDets = len(detIds)
        nmAnnos = len(annoIds)
        overLapData=[]

        # finding all overlaps based on IOU
        for idd in detIds:
            iouList=[]
            for ix in annoIds:
                iou =0
                iou=findIou(annoData[ix], detData[idd][0])
                #print (idd,ix,iou)
                iouList.append(iou)
            overLapData.append(iouList)


        #selecting the best match
        for i in range(nmDets):
            maxIou = 0;
            maxId = 0
            for j in range(nmAnnos):
                if(overLapData[i][j] > maxIou):
                    maxIou = overLapData[i][j]
                    maxId = j
            foundMatch =0;
            if(maxIou > 0):
                foundMatch =1

            for p in range(nmDets):
                if(overLapData[p][maxId] > maxIou):
                    foundMatch =0;
            if foundMatch ==1:
                legendData[annoIds[maxId]].append([detIds[i],maxIou])
                if(enableVisualizer):
                    visualiseBlended(detData[detIds[i]][0],annoData[annoIds[maxId]],maxIou)


        #writing to output file
    writeOutput(legendData,outputFileName)




if __name__== "__main__":
    main()


