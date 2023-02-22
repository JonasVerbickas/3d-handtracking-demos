import cv2 as cv
import numpy as np
from model import Model
import dearpygui.dearpygui as dpg
from view import View
from keypoint_estimator_enum import KeypointEstimatorEnum
from palm_detector_enum import PalmDetectorEnum

INITIAL_KEYPOINT_EST = KeypointEstimatorEnum.NONE
INITIAL_PALM_DET = PalmDetectorEnum.MEDIAPIPE

class GUIController:
    def __init__(self):
        self.model = Model(INITIAL_KEYPOINT_EST, INITIAL_PALM_DET)
        self.view = View(self.model.video_width, self.model.video_height,
                         self.keypoint_estimator_callback)
        self.render_loop()
        
    def keypoint_estimator_callback(self, sender, app_data, user_data):
        """
        Changes which inference method is used
        """
        self.model.load_keypoint_estimator(KeypointEstimatorEnum[app_data])
        dpg.configure_item(
            "palm_detector_combo", show=KeypointEstimatorEnum[app_data] == KeypointEstimatorEnum.MESHFORMER)

    def render_loop(self):
        """
        Blocking function
        """
        while dpg.is_dearpygui_running():
            frame = self.model.get_new_annotated_frame()
            self.view.update_frame(frame)
        self.model.cap.release()
        dpg.destroy_context()

if __name__ == '__main__':
    GUIController()