import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("/home/eyal/Documents/projects/python/code/Saved models/sign_lang_detector.h5",
                                   compile=True)

image = cv2.imread("stuff/asl_alphabet_test/asl_alphabet_test/D_test.jpg")

inp = np.array([image])

pred = model.predict(inp)

ans = pred[0]

maxIndex = 0
for i in range(len(ans)):
    if ans[i] > ans[maxIndex]:
        maxIndex = i

print(maxIndex)
