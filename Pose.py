# import mediapipe as mp
# import cv2

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(imgRGB)

#     if results.pose_landmarks:
#         mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

#     cv2.imshow("MediaPipe Pose", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Commented out code
# # import pandas as pd
# # import os
# # points = mpPose.PoseLandmark
# # path = "photo"
# # data = []
# # for p in points:
# #     x = str(p)[13:]
# #     data.append(x + "_x")
# #     data.append(x + "_y")
# #     data.append(x + "_z")
# #     data.append(x + "_vis")
# # data = pd.DataFrame(columns=["exercise", "step"] + data)
# # count = 0
# # for img in os.listdir(path):
# #     temp = []
# #     if not img.endswith(".png"):
# #         print(f"Skipping invalid file: {img}")
# #         continue
# #     if "_" in img:
# #         exercise, step = img.split("_")[:2]
# #     else:
# #         exercise = "default_exercise"
# #         step = count + 1
# #     img = cv2.imread(path + "/" + img)
# #     imageWidth, imageHeight = img.shape[:2]
# #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     im = img.copy()
# #     results = pose.process(imgRGB)
# #     if results.pose_landmarks:
# #         mpDraw.draw_landmarks(im, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
# #         landmarks = results.pose_landmarks.landmark
# #         for i, j in zip(points, landmarks):
# #             temp = temp + [j.x, j.y, j.z, j.visibility]
# #         temp = [exercise, int(step)] + temp
# #         data.loc[count] = temp
# #         count += 1
# #     cv2.imshow("Image", im)
# #     cv2.waitKey(1000)
# # data.to_csv("points.csv", index=False)