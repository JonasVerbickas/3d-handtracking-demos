from .gui_model import Model
from .gui_view import View
from consts.keypoint_estimator_enum import KeypointEstimatorEnum
import dearpygui.dearpygui as dpg

class GUI:
    def __init__(self):
        self.model = Model()
        self.view = View(self.model.video_width, self.model.video_height,
                         self.keypoint_estimator_callback)
        self.render_loop()

    def keypoint_estimator_callback(self, sender, app_data, user_data):
        """
        Changes which inference method is used
        """
        self.model.load_keypoint_estimator(KeypointEstimatorEnum[app_data])
        dpg.configure_item(
            "palm_detector_combo", show=self.model.keypoint_estimator.requires_detector)

    def render_loop(self):
        """
        Blocking function
        """
        while dpg.is_dearpygui_running():
            frame = self.model.get_new_annotated_frame()
            self.view.update_frame(frame)
        self.model.cap.release()
        dpg.destroy_context()
