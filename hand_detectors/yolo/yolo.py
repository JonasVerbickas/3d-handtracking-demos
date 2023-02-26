# Implementation taken from https://github.com/cansik/yolo-hand-detection
import time
import cv2
import numpy as np
from typing import Union

CONFIG_PATH = "/Users/jonas/git/3rdYearProject/HandTracking GUIs/hands_via_dear_py/hand_detectors/yolo/dependencies/cross-hands-tiny.cfg"
MODEL_PATH = "/Users/jonas/git/3rdYearProject/HandTracking GUIs/hands_via_dear_py/hand_detectors/yolo/dependencies/cross-hands-tiny.weights"


class YOLO:
    def __init__(self, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.output_names = []
        self.labels = ['hand']
        try:
            self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
        except:
            raise ValueError("Couldn't find the models!\nDid you forget to download them manually (and keep in the "
                             "correct directory, models/) or run the shell script?")

        ln = self.net.getLayerNames()
        for i in self.net.getUnconnectedOutLayers():
            self.output_names.append(ln[int(i) - 1])

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image):
        ih, iw = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.output_names)
        end = time.time()
        inference_time = end - start

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

        return iw, ih, inference_time, results
    
    def __call__(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        """
        For now it will track 1 hand only.
        """
        width, height, inference_time, results = self.inference(frame)
        # sort by confidence
        results.sort(key=lambda x: x[2])
        # how many hands should be shown
        if len(results) > 0:
            id, name, confidence, x, y, w, h = results[0]
            return frame[y:y+h, x:x+w]
        else:
            return None


if __name__ == '__main__':
    yolo = YOLO()
    print("starting webcam...")
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    rval, frame = vc.read()
    while rval:
        width, height, inference_time, results = yolo.inference(frame)
        # display fps
        cv2.putText(frame, f'{round(1/inference_time,2)} FPS',
                    (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # sort by confidence
        results.sort(key=lambda x: x[2])
        # how many hands should be shown
        hand_count = 1
        # display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow("preview")
    vc.release()
