# -*- coding: utf-8 -*-
from numpy import csingle
from ultralytics import YOLO
import os
import logging
import cv2


def face_detection_by_video():
    model = YOLO(os.path.join("models", "yolov8n-face.pt"))
    results = model.predict(source="demo.mp4", show=True)
    logging.info(results)


def face_detection_by_image():
    model = YOLO(os.path.join("models", "yolov8n-face.pt"))
    image_detection_file: str = os.path.join("provision", "examples", "face.jpeg")
    results = model(image_detection_file)
    boxes = results[0].boxes
    image_read = cv2.imread(image_detection_file)
    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0][0])
        top_left_y = int(box.xyxy.tolist()[0][1])
        bottom_right_x = int(box.xyxy.tolist()[0][2])
        bottom_right_y = int(box.xyxy.tolist()[0][3])

        cv2.rectangle(
            img=image_read,
            pt1=(top_left_x, top_left_y),
            pt2=(bottom_right_x, bottom_right_y),
            color=(50, 200, 129),
            thickness=2,
        )
        cv2.imwrite(os.path.join("build", "detection.jpeg"), image_read)


if __name__ == "__main__":
    face_detection_by_image()