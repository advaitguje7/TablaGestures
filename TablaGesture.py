import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)


    prev_pinched_pinky0 = False
    last_trigger_time = 0.0
    cooldown_s = 0.12

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(mp_image)

        status = "NO HAND"

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # MediaPipe landmark indices
            
            thumb = lm[4]

            pinky = []
            ring = []
            middle = []
            index = []
            fingers = [pinky, ring, middle, index]

            pinky0 = lm[17]
            pinky1 = lm[18]
            pinky2 = lm[19]
            pinky3 = lm[20]

            ring0 = lm[13]
            ring1 = lm[14]
            ring2 = lm[15]
            ring3 = lm[16]

            middle0 = lm[9]
            middle1 = lm[10]
            middle2 = lm[11]
            middle3 = lm[12]

            index0 = lm[5]
            index1 = lm[6]
            index2 = lm[7]
            index3 = lm[8]


            wrist = lm[0]


            thumb_xy = (int(thumb.x * w), int(thumb.y * h))

            # pinky
            pinky0_xy = (int(pinky0.x * w), int(pinky0.y * h))
            pinky1_xy = (int(pinky1.x * w), int(pinky1.y * h))
            pinky2_xy = (int(pinky2.x * w), int(pinky2.y * h))
            pinky3_xy = (int(pinky3.x * w), int(pinky3.y * h))

            # ring
            ring0_xy = (int(ring0.x * w), int(ring0.y * h))
            ring1_xy = (int(ring1.x * w), int(ring1.y * h))
            ring2_xy = (int(ring2.x * w), int(ring2.y * h))
            ring3_xy = (int(ring3.x * w), int(ring3.y * h))          

            # middle
            middle0_xy = (int(middle0.x * w), int(middle0.y * h))
            middle1_xy = (int(middle1.x * w), int(middle1.y * h))
            middle2_xy = (int(middle2.x * w), int(middle2.y * h))
            middle3_xy = (int(middle3.x * w), int(middle3.y * h))

            # index
            index0_xy = (int(index0.x * w), int(index0.y * h))
            index1_xy = (int(index1.x * w), int(index1.y * h))
            index2_xy = (int(index2.x * w), int(index2.y * h))
            index3_xy = (int(index3.x * w), int(index3.y * h))

            # thumb circle
            cv2.circle(frame, thumb_xy, 8, (0, 255, 0), -1)

            # pinky circles
            cv2.circle(frame, pinky0_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, pinky1_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, pinky2_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, pinky3_xy, 8, (0, 255, 0), -1)

            # ring circles
            cv2.circle(frame, ring0_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, ring1_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, ring2_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, ring3_xy, 8, (0, 255, 0), -1)

            # middle circles
            cv2.circle(frame, middle0_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, middle1_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, middle2_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, middle3_xy, 8, (0, 255, 0), -1)

            # index circles
            cv2.circle(frame, index0_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, index1_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, index2_xy, 8, (0, 255, 0), -1)
            cv2.circle(frame, index3_xy, 8, (0, 255, 0), -1)


            cv2.line(frame, thumb_xy, index3_xy, (0, 255, 0), 2) # line: thumb to index3

            pinch_px = dist(thumb_xy, pinky0_xy)
            

            pinch_px_matra1 = dist(thumb_xy, pinky0_xy)
            pinch_px_matra2 = dist(thumb_xy, pinky1_xy)
            pinch_px_matra3 = dist(thumb_xy, pinky2_xy)
            pinch_px_matra4 = dist(thumb_xy, pinky3_xy)

            pinch_px_matra5 = dist(thumb_xy, ring0_xy)
            pinch_px_matra6 = dist(thumb_xy, ring1_xy)
            pinch_px_matra7 = dist(thumb_xy, ring2_xy)
            pinch_px_matra8 = dist(thumb_xy, ring3_xy)

            pinch_px_matra9 = dist(thumb_xy, middle0_xy)
            pinch_px_matra10 = dist(thumb_xy, middle1_xy)
            pinch_px_matra11 = dist(thumb_xy, middle2_xy)
            pinch_px_matra12 = dist(thumb_xy, middle3_xy)

            pinch_px_matra13 = dist(thumb_xy, index0_xy)
            pinch_px_matra14 = dist(thumb_xy, index1_xy)
            pinch_px_matra15 = dist(thumb_xy, index2_xy)
            pinch_px_matra16 = dist(thumb_xy, pinky3_xy)


            PINCH_ON = 60
            PINCH_OFF = 90

            if prev_pinched_pinky0:
                pinched  = pinch_px_matra1 < PINCH_OFF
            else:
                pinched  = pinch_px_matra1 < PINCH_ON

            now = time.time()
            if pinched and not prev_pinched_pinky0 and (now - last_trigger_time > cooldown_s):
                last_trigger_time = now
                print("Matra 1 (Sam)")

            prev_pinched_pinky0 = pinched
            status = f"PINCH_PX: {pinch_px_matra1:.1f}"

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Hand V1 (new API)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
