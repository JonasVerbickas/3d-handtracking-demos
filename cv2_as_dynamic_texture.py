import dearpygui.dearpygui as dpg
import cv2 as cv
import numpy as np
from  mediapipe_inference import inference
from enum import Enum

class PalmDetector(Enum):
    MEDIAPIPE = 1
    YOLO = 2

class KeypointEstimator(Enum):
    NONE = 0
    MEDIAPIPE = 1
    MESHFORMER = 2

models_that_require_palm_detection = [KeypointEstimator.MESHFORMER]

class GUI:
    def __init__(self):
        self.keypoint_estimator = KeypointEstimator.NONE
        self.palm_detector = PalmDetector.MEDIAPIPE 
        video_width, video_height = self.init_camera()
        self.init_layout(video_width, video_height)
        self.render_loop()
    
    def init_camera(self):
        self.cap = cv.VideoCapture(0)
        ret, frame = self.cap.read()
        assert ret
        width, height = frame.shape[1], frame.shape[0]
        return width, height

    def init_layout(self, video_width, video_height):
        dpg.create_context()
        # dpg.configure_app(docking=True, docking_space=True)
        dpg.create_viewport(title="Hand Tracking Demo")
        dpg.setup_dearpygui()
        # dpg.set_global_font_scale(1.05)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(video_width, video_height, default_value=[], tag="frame")
        with dpg.window(label='Webcam Footage', tag="main_window", no_resize=True, no_title_bar=True,
                        no_move=True, no_collapse=True, no_bring_to_front_on_focus=True):
            dpg.add_image("frame")
        with dpg.window(label="Model Selector", width=300):
            dpg.add_combo([k.name for k in KeypointEstimator], default_value=str(self.keypoint_estimator.name),
                          width=250, tag='keypoint_estimator_combo', callback=self.keypoint_estimator_callback)
            dpg.add_combo([k.name for k in PalmDetector], default_value=str(self.palm_detector.name),
                          width=250, tag='palm_detector_combo', callback=self.keypoint_estimator_callback, show=False)
        dpg.show_viewport()
        dpg.set_primary_window('main_window', False)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        assert ret
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if self.keypoint_estimator == KeypointEstimator.MEDIAPIPE:
            frame = inference(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        frame = np.array(frame, dtype=np.float32).ravel()/255
        dpg.set_value("frame", frame)
        dpg.render_dearpygui_frame()
        
    
    def keypoint_estimator_callback(self, sender, app_data, user_data):
        """
        Changes which inference method is used
        """
        self.keypoint_estimator = KeypointEstimator[app_data]
        dpg.configure_item(
            "palm_detector_combo", show=self.keypoint_estimator == KeypointEstimator.MESHFORMER)


    def render_loop(self):
        """
        Blocking function
        """
        while dpg.is_dearpygui_running():
            self.update_frame()
        self.cap.release()
        dpg.destroy_context()

if __name__ == '__main__':
    GUI()