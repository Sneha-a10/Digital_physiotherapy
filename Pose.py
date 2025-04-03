import mediapipe as mp
import cv2
import pandas as pd
import os

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks
path = "C:/Users/Sneha/OneDrive/Desktop/college/gdg/photo/" # enter dataset path
data = []

for p in points:
        x = str(p)[13:]
        data.append(x + "_x")
        data.append(x + "_y")
        data.append(x + "_z")
        data.append(x + "_vis")
    
data = pd.DataFrame(columns = data)

count = 0

for img in os.listdir(path):
        temp = []
        img = cv2.imread(path + "/" + img)
        imageWidth, imageHeight = img.shape[:2]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = img.copy()
        results = pose.process(imgRGB)
        if results.pose_landmarks:

                mpDraw.draw_landmarks(im, results.pose_landmarks, mpPose.POSE_CONNECTIONS) 
                landmarks = results.pose_landmarks.landmark
                for i,j in zip(points,landmarks):
                    temp = temp + [j.x, j.y, j.z, j.visibility]
                data.loc[count] = temp
                count +=1

        cv2.imshow("Image", im)

        cv2.waitKey(1000)

data.to_csv("points.csv") 