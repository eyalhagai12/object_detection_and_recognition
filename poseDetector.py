import cv2
import mediapipe as mp

# capture webcam feed
cap = cv2.VideoCapture(0)

# initiate pose model
mp_pose_sel = mp.solutions.pose
mp_pose = mp_pose_sel.Pose()

# add drawing utils
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles.DrawingSpec((0, 0, 0))
mp_style2 = mp.solutions.drawing_styles.DrawingSpec((200, 200, 200))

# process video
while True:
    # get current image
    success, img = cap.read()

    # convert image to rgb from bgr
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # process using the pose model
    process = mp_pose.process(imgRGB)

    # draw the landmarks
    if process.pose_landmarks:
        mp_draw.draw_landmarks(img, process.pose_landmarks, mp_pose_sel.POSE_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
