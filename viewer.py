import cv2
import torch


def get_crop(frame, p1, p2):
    """
    Get the crop of the hand in the picture

    :param frame: the current frame
    :param p1: a point of the bounding box
    :param p2: another point of the bounding box
    :return: the cropped image of the hand, resized and in grayscale
    """
    c_image = frame[p1[1]:p2[1], p1[0]:p2[0]]
    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)
    c_image = cv2.resize(c_image, (224, 224))

    return c_image


class viewer:
    """
    This class if for showing the results of the 3 models used to detect, track and
    recognize the sign language
    """

    def __init__(self):
        self.controller = None
        self.labels = [chr(ord("A") + i) for i in range(26)] + ["Del", "Space", "Nothing"]

    def set_controller(self, controller):
        self.controller = controller

    def view(self):
        video = cv2.VideoCapture(-1)

        while True:
            # Read a new frame
            success, frame = video.read()
            if not success:
                break

            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, p1, p2 = self.controller.get_tracker_bbox(frame)

            # Calculate Frames per second (FPS)c
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok and success:
                # Tracking success
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                # view hand crop
                c_image = get_crop(frame, p1, p2)
                out = self.controller.process_sign(c_image)
                idx = torch.argmax(out[0]).item()
                print(idx)
                cv2.putText(frame, "Label : " + self.labels[idx], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (50, 170, 50), 2)
                print("Predicted Label: {}".format(self.labels[idx]))
                cv2.imshow("Crop", c_image)
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
