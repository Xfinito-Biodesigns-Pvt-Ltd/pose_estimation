# Library files:
import mediapipe as mdp
import math
import numpy as np
import time
import streamlit as st
import cv2

# Initial conditions:
prev_time = 0
steps = 0
step_left = 0
step_right = 0
stance = None
L = 0
H = 0
L1 = 0
H1 = 0


## For executing webapplication:
def main():
    """Pose estimation App"""


st.title("Pose estimator")

html_temp = """
          <body style="background-color:red;">
           <div style="background-color:teal ;padding:10px">
           <h2 style="color:white;text-align:center;">Pose Estimation WebApp</h2>
           </div>
           </body>
           """
st.markdown(html_temp, unsafe_allow_html=True)

video_file = st.file_uploader("Upload your video", type=['mp4', 'avi'])


## Main logic algorithm:
while True:
     ret, img = video_file.read()
     col_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

     pose_sols = mdp.solutions.pose.Pose().process(col_img)
     landmarks_info = pose_sols.pose_landmarks

     if landmarks_info:
         mdp.solutions.drawing_utils.draw_landmarks(img, landmarks_info,
                                                    mdp.solutions.pose.POSE_CONNECTIONS)
     landmarks_info_list = []
     for idx, landmarks in enumerate(landmarks_info.landmark):
         hgt, wdt, chnl = img.shape

         landmarks_info_list.append([idx, int(landmarks.x * wdt), int(landmarks.y * hgt)])
         cv2.circle(img, (int(landmarks.x * wdt), int(landmarks.y * hgt)), 3, (0, 255, 0))

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

         cv2.putText(img, str(int(lt_leg_angle)), (lt_knee25[0] - 20, lt_knee25[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                     1.5,
                     (0, 0, 255), 2)

         cv2.putText(img, str(int(rt_leg_angle)), (rt_knee26[0] - 20, rt_knee26[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                     1.5,
                     (0, 0, 255), 2)

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

     L = 0

     if (rt_leg_angle <= 15):
         L1 = 0
         H1 = 0

     if (rt_leg_angle >= 15 and H1 == 0):
         L1 = 1

     else:
         L1 = 0

     if rt_leg_angle >= 45 and L1 == 1:
         H1 = 1

     K = [L1 == 1 and H1 == 1]
     if True in K:
         step_right += 1

     L1 = 0

     if True in K or True in D:
         total_steps = step_left + step_right

         print(f'Step no:', {total_steps})

         # arr = []
         # for i in range(0, total_steps):
         # arr.append(total_steps)
         # print(arr)

     #cur_time = time.time()
     #fps = 1 / (cur_time - prev_time)
     #prev_time = cur_time

     #cv2.imshow("Pose estimator", img)


#st.video(img)
st.text(total_steps)
     
if __name__ == '__main__':
        main()
         








