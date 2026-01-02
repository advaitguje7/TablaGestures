import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import Enum

MODEL_PATH = "hand_landmarker.task"

# TO-DO
# Add dynamic list for multiple talas
# Add reset func
# get matras to work
# add dynamic threshold
# improve sensitivity
# fix display

def dist_3d(p1, p2, w, h):
    # Convert normalized MediaPipe coordinates to pixel/scale space
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    dz = (p1.z - p2.z) * w
    return math.sqrt(dx**2 + dy**2 + dz**2)
def dist_virtual(p1, p2_xy):
    """
    Docstring for dist_virtual
    
    :param p1: lm[]
    :param p2_xy: tuple
    """

    p2x = p2_xy[0]
    p2y = p2_xy[1]

    dx = (p1.x - p2x)
    dy = (p1.y - p2y)

    return math.sqrt(dx**2 + dy**2)
    

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
    prev_pinched_pinky1 = False
    prev_pinched_pinky2 = False
    prev_pinched_pinky3 = False

    prev_pinched_ring0 = False
    prev_pinched_ring1 = False
    prev_pinched_ring2 = False
    prev_pinched_ring3 = False

    prev_pinched_middle0 = False
    prev_pinched_middle1 = False
    prev_pinched_middle2 = False
    prev_pinched_middle3 = False

    prev_pinched_index0 = False
    prev_pinched_index1 = False
    prev_pinched_index2 = False
    prev_pinched_index3 = False

    prev_pinched = [prev_pinched_pinky0, prev_pinched_pinky1, prev_pinched_pinky2, prev_pinched_pinky3,
                    prev_pinched_ring0, prev_pinched_ring1, prev_pinched_ring2, prev_pinched_ring3,
                    prev_pinched_middle0, prev_pinched_middle1, prev_pinched_middle2, prev_pinched_middle3,
                    prev_pinched_index0, prev_pinched_index1, prev_pinched_index2, prev_pinched_index3]
    
    prev_pinched_init_0 = False
    prev_pinched_init_1 = False
    prev_pinched_init_2 = False
    prev_pinched_init_3 = False

    last_trigger_time = [0.0] * 16
    cooldown_s = 0.5

    class Finger(Enum):
                 MISSING = -1
                 PALM = 0
                 PINKY = 1
                 RING = 2
                 MIDDLE = 3
                 INDEX = 4

    select = Finger.MISSING

    Teentaal = ['Dha', 'Dhin', 'Dhin', 'Dha', 
            'Dha', 'Dhin', 'Dhin', 'Dha',
            'Dha', 'Tin', 'Tin', 'Ta',
            'Ta', 'Dhin', 'Dhin', 'Dha']
    Ektaal = ['Dhin', 'Dhin', 'DhaGe', 'Tirkit', 'Tu', 'Na',
              'Kat', 'Ta', 'DhaGe', 'Tirkit', 'Dhin', 'Na']
    Jhaptaal = ['Dhi', 'Na', 'Dhi', 'Dhi', 'Na',
                'Ti', 'Na', 'Dhi', 'Dhi', 'Na']
    Rupak = ['Ti', 'Ti', 'Na',
             'Dhin', 'Na', 'Dhin', 'Na']
    Taals = [Teentaal, Ektaal, Jhaptaal, Rupak]

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
            thumb_xy = (int(thumb.x * w), int(thumb.y * h))

            ########## INITIAL #################### 
            pinky3 = lm[20]
            pinky3_xy = (int(pinky3.x * w), int(pinky3.y * h))

            ring3 = lm[16]
            ring3_xy = (int(ring3.x * w), int(ring3.y * h))

            middle3 = lm[12]
            middle3_xy = (int(middle3.x * w), int(middle3.y * h))

            index3 = lm[8]
            index3_xy = (int(index3.x * w), int(index3.y * h))

            # wrist = lm[0]
            reset = lm[2]

            # center_xy = (int((((wrist.x  + reset.x) / 2) * w)), int((((wrist.y  + reset.y) / 2) * h)))
            reset_xy = (int(reset.x * w), int(reset.y * h))
            reset_dist = dist_virtual(thumb, reset_xy)

            # s = dist_3d(wrist, mid_mcp, w, h)
            # threshold = 0.025 * s
            cv2.circle(frame, thumb_xy, 5, (255, 0, 255), -1)
            cv2.circle(frame, reset_xy, 8, (255, 255, 0), -1)


            

            show_circle = True

            if show_circle & (select == Finger.MISSING):
                cv2.circle(frame, pinky3_xy, 5, (0, 0, 255), -1)
                cv2.circle(frame, ring3_xy, 5, (0, 255, 255), -1)
                cv2.circle(frame, middle3_xy, 5, (0, 255, 0), -1)
                cv2.circle(frame, index3_xy, 5, (255, 0, 0), -1)

            pinch_px_init_pinky = dist_3d(thumb, pinky3, w, h)
            pinch_px_init_ring = dist_3d(thumb, ring3, w, h)
            pinch_px_init_middle = dist_3d(thumb, middle3, w, h)
            pinch_px_init_index = dist_3d(thumb, index3, w, h)

            pinched_pinky  = pinch_px_init_pinky < 10
            pinched_ring = pinch_px_init_ring < 10
            pinched_middle = pinch_px_init_middle < 10
            pinched_index = pinch_px_init_index < 10
        
            now = time.time()
            if select == Finger.MISSING:
                if pinched_pinky and not prev_pinched_init_0 and (now - last_trigger_time[3] > cooldown_s):
                        last_trigger_time[3] = now
                        print("Pinky")
                        select = Finger.PINKY
                # prev_pinched_init_0 = pinched_pinky
                status = f"PINCH_PX: {pinch_px_init_pinky:.1f}"

                if pinched_ring and not prev_pinched_init_1 and (now - last_trigger_time[7] > cooldown_s):
                        last_trigger_time[7] = now
                        print("Ring")
                        select = Finger.RING

                # prev_pinched_init_1 = pinched_ring
                status = f"PINCH_PX: {pinch_px_init_ring:.1f}"
                
                if pinched_middle and not prev_pinched_init_2 and (now - last_trigger_time[10] > cooldown_s):
                        last_trigger_time[10] = now
                        print("Middle")
                        select = Finger.MIDDLE

                # prev_pinched_init_2 = pinched_middle
                status = f"PINCH_PX: {pinch_px_init_middle:.1f}"

                if pinched_index and not prev_pinched_init_3 and (now - last_trigger_time[13] > cooldown_s):
                        last_trigger_time[13] = now
                        print("Index")
                        select = Finger.INDEX

                # prev_pinched_init_3 = pinched_index
                status = f"PINCH_PX: {pinch_px_init_index:.1f}"

            ############### CONDITIONAL #####################################
            pinch_px = []
            offset = 0
            if select == Finger.PINKY:
                show_circle = False
                
                # pinky landmarks
                pinky0 = lm[17]
                pinky1 = lm[18]
                pinky0_cxy = (int(((pinky0.x + pinky1.x) / 2) * w), int(((pinky0.y + pinky1.y) / 2) * h))
                pinky2 = lm[19]

                # pinky_xy
                pinky0_xy = (int(pinky0.x * w), int(pinky0.y * h))
                pinky1_xy = (int(pinky1.x * w), int(pinky1.y * h))
                pinky2_xy = (int(pinky2.x * w), int(pinky2.y * h))
                
                # pinky circles
                cv2.circle(frame, pinky0_xy, 5, (0, 0, 255), -1)
                cv2.circle(frame, pinky1_xy, 5, (0, 0, 255), -1)
                cv2.circle(frame, pinky2_xy, 5, (0, 0, 255), -1)
                cv2.circle(frame, pinky3_xy, 5, (0, 0, 255), -1)

                #pinky lines
                # cv2.line(frame, thumb_xy, pinky0_xy, (255, 255, 255), 2)
                # cv2.line(frame, thumb_xy, pinky1_xy, (255, 255, 255), 2) 
                # cv2.line(frame, thumb_xy, pinky2_xy, (255, 255, 255), 2) 
                # cv2.line(frame, thumb_xy, pinky3_xy, (255, 255, 255), 2) 

                # pinky matras
                pinch_px_matra1 = dist_3d(thumb, pinky0, w, h)
                pinch_px_matra2 = dist_3d(thumb, pinky1, w, h)
                pinch_px_matra3 = dist_3d(thumb, pinky2, w, h)
                pinch_px_matra4 = dist_3d(thumb, pinky3, w, h)

                pinch_px = [pinch_px_matra1, pinch_px_matra2, pinch_px_matra3, pinch_px_matra4]
                
            elif select == Finger.RING:
                show_circle = False

                # ring landmarks
                ring0 = lm[13]
                ring1 = lm[14]
                ring0_cxy = (int(((ring0.x + ring1.x) / 2) * w),int(((ring0.y + ring1.y) / 2) * h))
                ring2 = lm[15]

                # ring_xy
                ring0_xy = (int(ring0.x * w), int(ring0.y * h))
                ring1_xy = (int(ring1.x * w), int(ring1.y * h))
                ring2_xy = (int(ring2.x * w), int(ring2.y * h))
                
                # ring circles
                cv2.circle(frame, ring0_xy, 5, (0, 255, 255), -1)
                cv2.circle(frame, ring1_xy, 5, (0, 255, 255), -1)
                cv2.circle(frame, ring2_xy, 5, (0, 255, 255), -1)
                cv2.circle(frame, ring3_xy, 5, (0, 255, 255), -1)

                # ring lines
                # cv2.line(frame, thumb_xy, ring0_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, ring1_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, ring2_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, ring3_xy, (0, 0, 0), 2) 

                # ring matras
                pinch_px_matra5 = dist_3d(thumb, ring0, w, h)
                pinch_px_matra6 = dist_3d(thumb, ring1, w, h)
                pinch_px_matra7 = dist_3d(thumb, ring2, w, h)
                pinch_px_matra8 = dist_3d(thumb, ring3, w, h)

                pinch_px = [pinch_px_matra5, pinch_px_matra6, pinch_px_matra7, pinch_px_matra8]

                offset = 4
                
            elif select == Finger.MIDDLE:
                show_circle = False

                # middle landmarks
                middle0 = lm[9]
                middle1 = lm[10]
                middle0_cxy = (int(((middle0.x + middle1.x) / 2) * w), int(((middle0.y + middle1.y) / 2) * h))
                middle2 = lm[11]

                # middle_xy
                middle0_xy = (int(middle0.x * w), int(middle0.y * h))
                middle1_xy = (int(middle1.x * w), int(middle1.y * h))
                middle2_xy = (int(middle2.x * w), int(middle2.y * h))

                # middle circles
                cv2.circle(frame, middle0_xy, 5, (0, 255, 0), -1)
                cv2.circle(frame, middle1_xy, 5, (0, 255, 0), -1)
                cv2.circle(frame, middle2_xy, 5, (0, 255, 0), -1)
                cv2.circle(frame, middle3_xy, 5, (0, 255, 0), -1)

                # middle lines
                # cv2.line(frame, thumb_xy, middle0_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, middle1_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, middle2_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, middle3_xy, (0, 0, 0), 2) 

                # middle matras
                pinch_px_matra9 = dist_3d(thumb, middle0, w, h)
                pinch_px_matra10 = dist_3d(thumb, middle1, w, h)
                pinch_px_matra11 = dist_3d(thumb, middle2, w, h)
                pinch_px_matra12 = dist_3d(thumb, middle3, w, h)

                pinch_px = [pinch_px_matra9, pinch_px_matra10, pinch_px_matra11, pinch_px_matra12]

                offset = 8
                
            elif select == Finger.INDEX:
                show_circle = False

                #index landmarks
                index0 = lm[5]
                index1 = lm[6]
                index0_cxy = (int(((index0.x + index1.x) / 2) * w), int(((index0.y + index1.y) / 2) * h))
                index2 = lm[7]
                index3 = lm[8]

                # index_xy
                index0_xy = (int(index0.x * w), int(index0.y * h))
                index1_xy = (int(index1.x * w), int(index1.y * h))
                index2_xy = (int(index2.x * w), int(index2.y * h))
                index3_xy = (int(index3.x * w), int(index3.y * h))

                #index circles
                cv2.circle(frame, index0_xy, 5, (255, 0, 0), -1)
                cv2.circle(frame, index1_xy, 5, (255, 0, 0), -1)
                cv2.circle(frame, index2_xy, 5, (255, 0, 0), -1)
                cv2.circle(frame, index3_xy, 5, (255, 0, 0), -1)    

                # index lines
                # cv2.line(frame, thumb_xy, index0_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, index1_xy, (0, 0, 0), 2)
                # cv2.line(frame, thumb_xy, index2_xy, (0, 0, 0), 2) 
                # cv2.line(frame, thumb_xy, index3_xy, (0, 0, 0), 2) 

                # index matras
                pinch_px_matra13 = dist_3d(thumb, index0, w, h)
                pinch_px_matra14 = dist_3d(thumb, index1, w, h)
                pinch_px_matra15 = dist_3d(thumb, index2, w, h)
                pinch_px_matra16 = dist_3d(thumb, index3, w, h)    

                pinch_px = [pinch_px_matra13, pinch_px_matra14, pinch_px_matra15, pinch_px_matra16]

                offset = 12
            
            if (reset_dist < 10):
                 select = Finger.MISSING
                 show_circle = True
            
            
            if (select != Finger.MISSING):
                 best = min(pinch_px)
            PINCH_ON = 30
            PINCH_OFF = 60
            
            for i in range(len(pinch_px)):
                idx = offset + i
                if (prev_pinched[idx] == True):  
                    pinched = pinch_px[i] < PINCH_OFF and (pinch_px[i] == best)
                else: 
                     pinched = pinch_px[i] < PINCH_ON and (pinch_px[i] == best)

                now = time.time()
                if (pinched == True) and (not prev_pinched[idx]) and (now - last_trigger_time[idx] > cooldown_s):
                    last_trigger_time[i] = now
                    cout = str(Teentaal[idx]) + ' ' + str(idx + 1)
                    if (idx == 0):
                         cout += ' (Sam)'
                    elif (idx == 8):
                         cout += ' Khali'
                    print(cout)

                prev_pinched[idx] = pinched
                status = f"PINCH_PX: {pinch_px[i]:.1f}"
            

        cv2.putText(frame, status, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imshow("Hand V1 (new API)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
