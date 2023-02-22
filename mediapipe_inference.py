import mediapipe as mp
import numpy as np

class MediaPipeJointEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
                model_complexity=1,
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) 

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return image