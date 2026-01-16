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

points_index = [
  1,    # nose tip
  152,  # chin
  33,   # left_eye left_corner
  263,  # right_eye right_corner
  61,   # left mouth corner
  291   # right mouth corner
]

face_3d_coords = np.array([
  (0.0, 0.0, 0.0),             # Nose tip
  (0.0, -330.0, -65.0),        # Chin
  (-225.0, 170.0, -135.0),     # Left eye left corner
  (225.0, 170.0, -135.0),      # Right eye right corner
  (-150.0, -150.0, -125.0),    # Left Mouth corner
  (150.0, -150.0, -125.0)      # Right mouth corner
])

def draw_camera_axis(img, axis_index, start, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
  end_point = [0.0, 0.0, 0.0]
  end_point[axis_index] = 500.0
  color = [0, 0, 0]
  color[axis_index] = 255
  axis_text= ["x-axis", "y-axis", "z-axis"]

  (end_point2D, jacobian) = cv2.projectPoints(np.array([end_point]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
  end = (int(end_point2D[0][0][0]), int(end_point2D[0][0][1]))
  cv2.line(img, start, end, color, 2)
  cv2.putText(img, axis_text[axis_index], (int(end_point2D[0][0][0])-10, int(end_point2D[0][0][1])-10), cv2.FONT_HERSHEY_PLAIN, 1, color)
  return img

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
        cv2.circle(frame_rgb, (x, y), 2, (0, 255, 0), -1)
        face_2d_coords.append((x, y))
      
      face_2d_coords = np.array(face_2d_coords, dtype="double")
      (success, rotation_vector, translation_vector) = cv2.solvePnP(face_3d_coords, face_2d_coords, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
      
      axis_origin = ( int(face_2d_coords[0][0]), int(face_2d_coords[0][1]))
      frame_rgb = draw_camera_axis(frame_rgb, 0, axis_origin, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
      frame_rgb = draw_camera_axis(frame_rgb, 1, axis_origin, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
      frame_rgb = draw_camera_axis(frame_rgb, 2, axis_origin, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    final_frame = cv2.hconcat([frame, annotated])

    cv2.imshow('webcam', final_frame)
    frame_index += 1
    # if you press 'm' of michael, stop displaying
    if cv2.waitKey(1) == ord('m'):
      break

  cap.release()
  cv2.destroyAllWindows()