from .gui_model import Model
from .gui_view import View
from consts.keypoint_estimator_enum import KeypointEstimatorEnum
from consts.palm_detector_enum import PalmDetectorEnum
import dearpygui.dearpygui as dpg

class GUI:
    def __init__(self):
        self.model = Model()
        self.view = View(self.model.video_width, self.model.video_height,
                         self.keypoint_estimator_callback, self.hand_det_callback)
        self.render_loop()

    def keypoint_estimator_callback(self, sender, app_data, user_data):
        """
        Changes which pose estimation method is used
        """
        self.model.load_keypoint_estimator(KeypointEstimatorEnum[app_data])
        dpg.configure_item(
            "keypoint_estimator_combo")
        dpg.configure_item(
            "palm_detector_combo", show=self.model.keypoint_estimator.requires_detector)

    def hand_det_callback(self, sender, app_data, user_data):
        """
        Changes which hand detection method is used
        """
        self.model.load_palm_detector(PalmDetectorEnum[app_data])

    def render_loop(self):
        """
        Infinite loop while the program is running.
        """
        while dpg.is_dearpygui_running():
            frame, cropped = self.model.get_new_annotated_frame()
            self.view.update_main_frame(frame)
            if cropped is not None:
                self.view.update_cropped_frame(cropped)
        self.model.cap.release()
        dpg.destroy_context()
