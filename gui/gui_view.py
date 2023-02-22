import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from consts.keypoint_estimator_enum import KeypointEstimatorEnum
from consts.palm_detector_enum import PalmDetectorEnum

class View:
    def __init__(self, video_width, video_height, keypoint_estimator_callback):
        dpg.create_context()
        # dpg.configure_app(docking=True, docking_space=True)
        dpg.create_viewport(title="Hand Tracking Demo")
        dpg.setup_dearpygui()
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(video_width, video_height,
                                default_value=[], tag="frame")
        with dpg.window(label='Webcam Footage', tag="main_window", no_resize=True, no_title_bar=True,
                        no_move=True, no_collapse=True, no_bring_to_front_on_focus=True):
            dpg.add_image("frame")
        with dpg.window(label="Model Selector", width=300):
            dpg.add_combo([k.name for k in KeypointEstimatorEnum], default_value=str("Please select an estimator you'd like to use"),
                          width=250, tag='keypoint_estimator_combo', callback=keypoint_estimator_callback)
            dpg.add_combo([k.name for k in PalmDetectorEnum], default_value=str(list(PalmDetectorEnum)[0]),
                          width=250, tag='palm_detector_combo', callback=keypoint_estimator_callback, show=False)
        dpg.show_viewport()
        dpg.set_primary_window('main_window', False)

    def update_frame(self, new_frame):
        frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2RGBA)
        frame = np.array(frame, dtype=np.float32).ravel()/255
        dpg.set_value("frame", frame)
        dpg.render_dearpygui_frame()
