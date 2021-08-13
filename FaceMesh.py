import cv2
import mediapipe as mp

# capture webcam feed
cap = cv2.VideoCapture(0)

# initiate model
mp_face_mesh = mp.solutions.face_mesh
model = mp_face_mesh.FaceMesh()

# add drawing utils
draw_util = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processed_image = model.process(img_rgb)

    if processed_image.multi_face_landmarks:
        for landmark in processed_image.multi_face_landmarks:
            draw_util.draw_landmarks(img, landmark, mp_face_mesh.FACE_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
