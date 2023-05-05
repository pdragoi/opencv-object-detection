import cv2
import time
from typing import Union
from threading import Thread
from models import *
import asyncio
import numpy as np


class ThreadedCamera(object):
    def __init__(
        self, 
        src: Union[str, int] = 0
    ):
        self.src = src
        self.capture = cv2.VideoCapture(
            src, 
        )
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(3, 1280)
        self.capture.set(4, 720)

        while not self.capture.isOpened():
            self.capture = cv2.VideoCapture(
                src, 
            )
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            time.sleep(0.1)


        self.model1 = EfficientDetD0()
        self.model2 = YOLOv3()

        self.model1 = cv2.dnn.readNet(self.model1.get_weights(), self.model1.get_config())
        self.model2 = cv2.dnn.readNet(self.model2.get_weights(), self.model2.get_config())
        # self.model1.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.model2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        
        # Set up image display windows
        self.win1 = "EfficientDet0 Output"
        self.win2 = "YOLOv3 Output"
        cv2.namedWindow(self.win1)
        cv2.namedWindow(self.win2)


        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        # self.FPS = 1/self.capture.get(cv2.CAP_PROP_FPS) / 1000
        # self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.start_capture, args=())
        self.thread.daemon = True
        self.thread.start()

    
    def start_capture(self):
        asyncio.run(self.update())
        

    async def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                if not self.status:
                    break

                # # Apply object detection to the frame using model 1 (EfficientDet5)
                task1 = asyncio.create_task(self.process_frame(self.model1, self.frame, (512, 512)))

                # # Apply object detection to the frame using model 2 (YOLOv2)
                task2 = asyncio.create_task(self.process_frame(self.model2, self.frame, (416, 416)))

                # Wait for both tasks to complete before continuing to next frame
                result = await asyncio.gather(task1, task2)
                
                ### model 2
                boxes, confidences = result[0]

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

                self.frame_model2 = self.frame.copy()

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image

                        # color = [int(c) for c in COLORS[class_ids[i]]]
                        cv2.rectangle(self.frame_model2, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
                        # text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
                        text = f"{confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = self.frame_model2.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(0, 255, 0), thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        self.frame_model2 = cv2.addWeighted(overlay, 0.6, self.frame_model2, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(self.frame_model2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 0), thickness=1)
                        

                 ### model 1
                boxes, confidences = result[0]

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

                self.frame_model1 = self.frame.copy()

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image

                        # color = [int(c) for c in COLORS[class_ids[i]]]
                        cv2.rectangle(self.frame_model1, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
                        # text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
                        text = f"{confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = self.frame_model1.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(0, 255, 0), thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        self.frame_model1 = cv2.addWeighted(overlay, 0.6, self.frame_model1, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(self.frame_model1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 0), thickness=1)


            else:
                self.capture = cv2.VideoCapture(
                    self.src, 
                )

            time.sleep(self.FPS)
            

    async def process_frame(self, model, frame, size):
        h = frame.shape[0]
        w = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, size=size, swapRB=True, crop=False)
        model.setInput(blob)
        outs = model.forward()
        # Process the detections and draw bounding boxes
        # ...
        boxes = []
        confidences = []
        lastLayer = model.getLayer(model.getLayerId(model.getLayerNames()[-1]))

        if lastLayer.type == 'Region':
            for detection in outs:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))


        elif lastLayer.type == 'DetectionOutput':
            for detection in outs[0, 0, :, :]:
                confidence = detection[2]
                if confidence > 0.5:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1

                    if width <= 2 or height <= 2:
                        left = int(detection[3] * w)
                        top = int(detection[4] * h)
                        right = int(detection[5] * w)
                        bottom = int(detection[6] * h)
                        width = right - left + 1
                        height = bottom - top + 1

                    boxes.append([left, top, int(width), int(height)])
                    confidences.append(float(confidence))

        return boxes, confidences


    def show_frame(self):
        cv2.imshow('stream', cv2.resize(self.frame, (0,0), fx=0.5, fy=0.5))
        cv2.imshow(self.win1, cv2.resize(self.frame_model1, (0,0), fx=0.5, fy=0.5))
        cv2.imshow(self.win2, cv2.resize(self.frame_model2, (0,0), fx=0.5, fy=0.5))

        cv2.waitKey(self.FPS_MS)


if __name__ == '__main__':
    ### use webcam for testing
    threaded_camera = ThreadedCamera(0)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass