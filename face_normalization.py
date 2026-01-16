"""
Copyright (c) 2026 hayunjong83@gmail.com
This work is a derivative of "Data-Normalization-Gaze-Estimation" by Xucong Zhang, used under CC BY-NC-SA 4.0.

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018

This Repository is licensed under CC BY-NC-SA 4.0 by hayunjong83.
However, this project is not intended for public distribution, so no additional license has been specified. 
For any questions, please refer to the original author
"""

import cv2
import os
import numpy as np
from urllib.request import urlretrieve
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dir_path = "landmark_model"
model_name = "face_landmarker_v2_with_blendshapes.task"
model_asset_path = os.path.join(dir_path, model_name)
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def model_check():
  if not os.path.isdir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

  if not os.path.exists(model_asset_path):
    urlretrieve(model_url, model_asset_path)
    print("face mesh file download")
  
  print(f"model is loaded : {model_asset_path}")

def model_load():
  BaseOptions = mp.tasks.BaseOptions
  FaceLandmarker = mp.tasks.vision.FaceLandmarker
  FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_asset_path),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
  )
  detector = FaceLandmarker.create_from_options(options)
  return detector

# This time, I used 'canonical face model' which is used with Mediapipe.
# ref link) https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
# Or, you can use 3D face model for MediaPipe 468 points mark-up.
# ref link) https://github.com/hysts/pytorch_mpiigaze_demo/blob/master/ptgaze/common/face_model_mediapipe.py#L9

points_index = [
  1,    # nose tip
  152,  # chin
  33,   # left_eye left_corner
  133,  # left_eye right corner

  362,  # right_eye left_corner
  263,  # right_eye right_corner

  61,   # left mouth corner
  291   # right mouth corner
]

face_3d_coords = np.array([
  (0.000000, -3.406404, 5.979507),      # Nose tip
  (0.000000, 6.545390, 5.027311),       # Chin

  (-2.266659, -7.425768, 4.389812),     # Left eye left corner
  (-7.270895, -2.890917, -2.252455),    # Left eye right corner

  (7.270895, -2.890917, -2.252455),     # Right eye left corner
  (2.266659, -7.425768, 4.389812),      # Right eye right corner
  (-0.729766, -1.593712, 5.833208),    # Left Mouth corner
  (0.729766, -1.593712, 5.833208)      # Right mouth corner
])

def estimate_head_pose(face_3d_coords, face_2d_coords, camera_matrix, dist_coeffs, iterate=True):
  (success, rvec, tvec) = cv2.solvePnP(face_3d_coords, face_2d_coords, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

  if iterate:
    success, rvec, tvec = cv2.solvePnP(face_3d_coords, face_2d_coords, camera_matrix, dist_coeffs, rvec, tvec, True)
  
  return rvec, tvec

def normalize_face(img, face_3d_coords, head_rvec, head_tvec, camera_matrix):
  focal_norm = 960
  distance_norm = 200
  roi_size = (224, 224)

  ht = head_tvec.reshape((3,1))
  hR = cv2.Rodrigues(head_rvec)[0]
  face_3d_camera_coords = np.dot(hR, face_3d_coords) + ht
  two_eye_center = np.mean(face_3d_camera_coords[:, 2:7], axis=1).reshape((3,1))
  nose_tip = face_3d_camera_coords[:, 0].reshape((3,1))
  face_center = np.mean(np.concatenate((two_eye_center, nose_tip), axis=1), axis=1).reshape((3,1))
  distance = np.linalg.norm(face_center)
  z_scale = distance_norm / distance
  cam_norm = np.array([
    [focal_norm, 0, roi_size[0]/2],
    [0, focal_norm, roi_size[1]/2],
    [0, 0., 1.0],
  ])
  S = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, z_scale]
  ])
  hRx = hR[:, 0]
  forward = (face_center / distance).reshape(3)
  down = np.cross(forward, hRx)
  down /= np.linalg.norm(down)
  right = np.cross(down, forward)
  right /= np.linalg.norm(right)
  R = np.c_[right, down, forward].T

  W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))

  img_warped = cv2.warpPerspective(img, W, roi_size)

  return img_warped
  
if __name__ == "__main__":
  model_check()
  detector = model_load()
  
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("cannot open webcam")
    exit()
  
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_index = 0

  # approximate camera intrinsic
  focal_length = width
  center = (height/2, width/2)
  camera_matrix = np.array(
    [
      [focal_length, 0, center[0]],
      [0, focal_length, center[1]],
      [0, 0, 1]
    ], dtype = "double"
  )
  dist_coeffs = np.zeros((4, 1))

  while True:
    ret, frame = cap.read()
    if not ret:
      print("cannot receive frame. Exit")
      break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)    
    frame_timestamp_ms = int(1000 * frame_index / fps)
    face_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    for face in face_landmarker_result.face_landmarks:
      face_2d_coords = []
      for idx in points_index:
        x = int(face[idx].x *width)
        y = int(face[idx].y * height)
        # cv2.circle(frame_rgb, (x, y), 2, (0, 255, 0), -1)
        face_2d_coords.append((x, y))
      break

    num_pts = face_3d_coords.shape[0]
    face_pts = face_3d_coords.reshape(num_pts, 1, 3)
    landmarks = np.array(face_2d_coords, dtype="float")
    landmarks = landmarks.reshape(num_pts, 1, 2)

    head_rvec, head_tvec = estimate_head_pose(face_pts, landmarks, camera_matrix, dist_coeffs)
    normalized_face = normalize_face(frame, face_3d_coords.T, head_rvec, head_tvec, camera_matrix)
    normalized_h, normalized_w, _ = normalized_face.shape
    final_frame = frame.copy()
    final_frame[0: normalized_h, 0: normalized_w] = normalized_face
    
    cv2.imshow('webcam', final_frame)
    frame_index += 1
    # if you press 'm' of michael, stop displaying
    if cv2.waitKey(1) == ord('m'):
      break

  cap.release()
  cv2.destroyAllWindows()