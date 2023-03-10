import cv2
import numpy as np
from typing import Tuple
# consts
import consts.intial_values as intial_values   
from consts.palm_detector_enum import PalmDetectorEnum
from consts.keypoint_estimator_enum import KeypointEstimatorEnum
# hand detectors
from hand_detectors.yolo.yolo import YOLO
from hand_detectors.blazepalm.blazepalm import BlazePalm
from pose_estimators.meshformer import MeshFormer
# keypoint estimators
from pose_estimators.identity import Identity
from pose_estimators.mediapipe_estimator import MediaPipeE2E

class Model:
    def __init__(self):
        self.load_keypoint_estimator(intial_values.KEYPOINT_EST)
        assert self.keypoint_estimator is not None
        self.load_palm_detector(intial_values.PALM_DET)
        assert self.palm_detector is not None
        self.video_width, self.video_height = self.init_camera()
    
    def load_keypoint_estimator(self, keypoint_estimator: KeypointEstimatorEnum) -> None:
        if keypoint_estimator == KeypointEstimatorEnum.MEDIAPIPE:
            self.keypoint_estimator = MediaPipeE2E()
        elif keypoint_estimator == KeypointEstimatorEnum.MESHFORMER:
            self.keypoint_estimator = MeshFormer()
        else:
            self.keypoint_estimator = Identity()
    
    def load_palm_detector(self, palm_detector: PalmDetectorEnum) -> None:
        if palm_detector == PalmDetectorEnum.YOLO:
            self.palm_detector = YOLO()
        else:
            self.palm_detector = BlazePalm()

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

    def get_new_annotated_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves new frame from webcam.
        Runs the selected model on that frame.
        Returns frame with keypoints drawn on it.
        """
        # 1. get a new video frame
        ret, frame = self.cap.read()
        assert ret
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. if model requires it crop a hand before estimation
        if self.keypoint_estimator.requires_detector:
            hand_bbox = self.palm_detector(frame)
            if hand_bbox is None:
                return frame, None
            # print('bbox', hand_bbox)
            # 2.1 cut out the bbox
            Mtr = cv2.getAffineTransform(
                        np.array(hand_bbox[:3]).astype(np.float32),
                        np.array([[0,0], [255, 0], [255, 255]]).astype(np.float32))
            warped_img = cv2.warpAffine(frame, Mtr, (frame.shape[1], frame.shape[0]))
            detected_hand = (warped_img[:256, :256, :]).copy()
        else:
            detected_hand = frame
        # 3. project cropped estimation back on the image
        image_with_pred = self.keypoint_estimator(detected_hand)
        if self.keypoint_estimator.requires_detector:
            warped_img[:256, :256, :] = image_with_pred
            iMtr = cv2.invertAffineTransform(Mtr)
            image_with_pred = cv2.warpAffine(warped_img, iMtr, (frame.shape[1], frame.shape[0]))
        # 4. return both the cropped result and estimation
        return image_with_pred, detected_hand
        

        