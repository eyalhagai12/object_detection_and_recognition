import cv2
import mediapipe as mp


class Objectron:
    def __init__(self, static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_name="Shoe", focal_length=(1.0, 1.0), principal_point=(0.0, 0.0),
                 image_size=None):
        self.mp_objectron = mp.solutions.objectron
        self.model = self.mp_objectron.Objectron(static_image_mode, max_num_objects, min_detection_confidence,
                                                 min_tracking_confidence, model_name, focal_length, principal_point,
                                                 image_size)
        self.draw_utils = mp.solutions.drawing_utils

    def process(self, img, draw=True):
        # convert from bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the image
        processed_image = self.model.process(img_rgb)

        if draw and processed_image.detected_objects:
            for obj in processed_image.detected_objects:
                self.draw_utils.draw_landmarks(img, obj.landmarks_2d, self.mp_objectron.BOX_CONNECTIONS)