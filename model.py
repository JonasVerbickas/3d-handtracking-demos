import cv2
import numpy as np
from typing import Tuple
from mediapipe_inference import MediaPipeJointEstimator
from keypoint_estimator_enum import KeypointEstimatorEnum
from palm_detector_enum import PalmDetectorEnum


class Model:
    def __init__(self, keypoint_estimator: KeypointEstimatorEnum, palm_detector: PalmDetectorEnum):
        self.keypoint_estimator = self.load_keypoint_estimator(keypoint_estimator)
        self.palm_detector: PalmDetectorEnum = palm_detector
        self.video_width, self.video_height = self.init_camera()
    
    def load_keypoint_estimator(self, keypoint_estimator_enum: KeypointEstimatorEnum):
        return MediaPipeJointEstimator()

    def init_camera(self) -> Tuple[int,int]:
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        assert ret
        width, height = frame.shape[1], frame.shape[0]
        return width, height

    def get_new_annotated_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        assert ret
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.keypoint_estimator(frame)
        return frame
        

        