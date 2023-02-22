import cv2
import numpy as np
from typing import Tuple
from pose_estimators.mediapipe_estimator import MediaPipeJointEstimator
from consts.keypoint_estimator_enum import KeypointEstimatorEnum
from consts.palm_detector_enum import PalmDetectorEnum
import consts.intial_values as intial_values   

class Model:
    def __init__(self):
        self.load_keypoint_estimator(intial_values.KEYPOINT_EST)
        self.load_palm_detector(intial_values.PALM_DET)
        # self.palm_detector: PalmDetectorEnum = palm_detector
        self.video_width, self.video_height = self.init_camera()
    
    def load_keypoint_estimator(self, keypoint_estimator: KeypointEstimatorEnum):
        if keypoint_estimator == KeypointEstimatorEnum.MEDIAPIPE:
            self.keypoint_estimator = MediaPipeJointEstimator()
        else:
            self.keypoint_estimator = lambda img: img
    
    def load_palm_detector(self, palm_detector: PalmDetectorEnum):
        return None

    def init_camera(self) -> Tuple[int,int]:
        """
        Initilizes the Camera.
        Returns its capture (width, height) -> useful when initializing windows.
        """
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        assert ret
        width, height = frame.shape[1], frame.shape[0]
        return width, height

    def get_new_annotated_frame(self) -> np.ndarray:
        """
        Retrieves new frame from webcam.
        Runs the selected model on that frame.
        Returns frame with keypoints drawn on it.
        """
        ret, frame = self.cap.read()
        assert ret
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.keypoint_estimator(frame)
        return frame
        

        