import cv2
import mediapipe as mdp
import math
import numpy as np
import time
import moviepy



mdp_drawing = mdp.solutions.drawing_utils
mdp_pose = mdp.solutions.pose

# In[2]:


vid_cap = cv2.VideoCapture('video5.mp4')
prev_time = 0
steps = 0
step_left = 0
step_right = 0
stance = None
L = 0
H = 0
L1 = 0
H1 = 0

with mdp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as pose:
 while  vid_cap.isOpened():
    ret, img = vid_cap.read()

    if ret:
        col_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting BGR2GRAY for the sake of mediapipe

        pose_sols = mdp.solutions.pose.Pose().process(col_img)  # mediapipe solutions for Pose detection
        landmarks_info = pose_sols.pose_landmarks  # to understand the landmarks

        if landmarks_info:
            mdp.solutions.drawing_utils.draw_landmarks(img, landmarks_info,
                                                   mdp.solutions.pose.POSE_CONNECTIONS)
        landmarks_info_list = []
        for idx, landmarks in enumerate(landmarks_info.landmark):  # identifying
            hgt, wdt, chnl = img.shape  # landmarks
        # at the their
            landmarks_info_list.append([idx, int(landmarks.x * wdt), int(landmarks.y * hgt)])
            cv2.circle(img, (int(landmarks.x * wdt), int(landmarks.y * hgt)), 3, (0, 255, 0))  # positions with a circle
        # print(landmarks_info_list)

    # lt_shoulder11 = [landmarks_info_list[11][1], landmarks_info_list[11][2]]
    # lt_elbow13 = [landmarks_info_list[13][1], landmarks_info_list[13][2]]
    # lt_wrist15 = [landmarks_info_list[15][1], landmarks_info_list[15][2]]

    # lt_hand_angle = abs(round(math.degrees(
    # math.atan2(lt_wrist15[1] - lt_elbow13[1], lt_wrist15[0] - lt_elbow13[0]) - math.atan2(
    # lt_shoulder11[1] - lt_elbow13[1], lt_shoulder11[0] - lt_elbow13[0]))))

    # rt_shoulder12 = [landmarks_info_list[12][1], landmarks_info_list[12][2]]
    # rt_elbow14 = [landmarks_info_list[14][1], landmarks_info_list[14][2]]
    # rt_wrist16 = [landmarks_info_list[16][1], landmarks_info_list[16][2]]

    # rt_hand_angle = abs(round(math.degrees(
    # math.atan2(rt_wrist16[1] - rt_elbow14[1], rt_wrist16[0] - rt_elbow14[0]) - math.atan2(
    # rt_shoulder12[1] - rt_elbow14[1], rt_shoulder12[0] - rt_elbow14[0]))))

        lt_hip23 = [landmarks_info_list[23][1], landmarks_info_list[23][2]]
        lt_knee25 = [landmarks_info_list[25][1], landmarks_info_list[25][2]]
        lt_ankle27 = [landmarks_info_list[27][1], landmarks_info_list[27][2]]
        lt_heel29 = [landmarks_info_list[29][1], landmarks_info_list[29][2]]
        lt_foot_index31 = [landmarks_info_list[31][1], landmarks_info_list[31][2]]

        lt_leg_angle = abs(180 - round(math.degrees(
        math.atan2(lt_ankle27[1] - lt_knee25[1], lt_ankle27[0] - lt_knee25[0]) - math.atan2(
            lt_hip23[1] - lt_knee25[1],
            lt_hip23[0] - lt_knee25[
                0]))))

        lt_ankle_angle = abs(360 - round(math.degrees(
        math.atan2(lt_foot_index31[1] - lt_ankle27[1], lt_foot_index31[0] - lt_ankle27[0]) - math.atan2(
            lt_knee25[1] - lt_ankle27[1],
            lt_knee25[0] - lt_ankle27[0]))))

        lt_angle = lt_ankle_angle

        rt_hip24 = [landmarks_info_list[24][1], landmarks_info_list[24][2]]
        rt_knee26 = [landmarks_info_list[26][1], landmarks_info_list[26][2]]
        rt_ankle28 = [landmarks_info_list[28][1], landmarks_info_list[28][2]]
        rt_heel30 = [landmarks_info_list[30][1], landmarks_info_list[30][2]]
        rt_foot_index32 = [landmarks_info_list[32][1], landmarks_info_list[32][2]]

        rt_leg_angle = abs(180 - round(math.degrees(
            math.atan2(rt_ankle28[1] - rt_knee26[1], rt_ankle28[0] - rt_knee26[0]) - math.atan2(
            rt_hip24[1] - rt_knee26[1],
            rt_hip24[0] - rt_knee26[
                0]))))

        if len(landmarks_info_list) != 0:
            cv2.circle(img, (rt_hip24[0], rt_hip24[1]), 5, (255, 0, 0))
            cv2.circle(img, (rt_knee26[0], rt_knee26[1]), 5, (255, 0, 0))
            cv2.circle(img, (rt_ankle28[0], rt_ankle28[1]), 5, (255, 0, 0))
            cv2.circle(img, (rt_heel30[0], rt_heel30[1]), 5, (255, 0, 0))
            cv2.circle(img, (rt_foot_index32[0], rt_foot_index32[1]), 5, (255, 0, 0))

            cv2.circle(img, (lt_hip23[0], lt_hip23[1]), 5, (255, 0, 0))
            cv2.circle(img, (lt_knee25[0], lt_knee25[1]), 5, (255, 0, 0))
            cv2.circle(img, (lt_ankle27[0], lt_ankle27[1]), 5, (255, 0, 0))
            cv2.circle(img, (lt_heel29[0], lt_heel29[1]), 5, (255, 0, 0))
            cv2.circle(img, (lt_foot_index31[0], lt_foot_index31[1]), 5, (255, 0, 0))

            #cv2.putText(img, str(int(lt_leg_angle)), (lt_knee25[0] - 20, lt_knee25[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                    #1.5,
                    #(0, 0, 255), 2)
        # cv2.putText(img, str(int(lt_angle)), (lt_ankle27[0] - 20, lt_ankle27[1] + 20), cv2.FONT_HERSHEY_PLAIN,
        #           1.5,
        #          (0, 0, 255), 2)
        # cv2.putText(img, str(int(rt_leg_angle)), (rt_knee26[0] - 20, rt_knee26[1] + 20), cv2.FONT_HERSHEY_PLAIN,
        #  1.5,
        # (0, 0, 255), 2)

        if (lt_leg_angle <= 15):
            L = 0
            H = 0

        if (lt_leg_angle >= 15 and H == 0):
            L = 1

        else:
            L = 0

        if lt_leg_angle >= 50 and L == 1:
            H = 1

        D = [L == 1 and H == 1]
        if True in D:
            step_left += 1
        # print(f'step_count', {step_left})
        # print(L, H)

        L = 0

    # Right leg angle:
        if (rt_leg_angle <= 10):
            L1 = 0
            H1 = 0

        if (rt_leg_angle >= 10 and H1 == 0):
            L1 = 1

        else:
            L1 = 0

        if rt_leg_angle >= 35 and L1 == 1:
            H1 = 1

        # print((rt_leg_angle))

        K = [L1 == 1 and H1 == 1]
        if True in K:
            step_right += 1
        # print(f'step_count_right', {step_right})
        # print(L, H)

        L1 = 0

        if True in K or True in D:
            total_steps = step_left + step_right

            print(f'Step no:', {total_steps})

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

        #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

        cv2.imshow("Video Frame", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

 vid_cap.release()
 cv2.destroyAllWindows()









