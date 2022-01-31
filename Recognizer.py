from myModel import myModel
from tensorflow import keras
import numpy as np


class recognizer(myModel):
    """
    A wrapper for the recognition model i trained to recognize letters in the american alphabet using
    sign language.
    """

    def __init__(self):
        self.model = keras.models.load_model("weights/best_sign_lang_model/best_sign_lang_model")

    def process(self, image):
        return self.model.predict(np.expand_dims(np.expand_dims(image, -1), 0)).argmax()
