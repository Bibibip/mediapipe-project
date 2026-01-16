import cv2
import os
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
    )
    detector = FaceLandmarker.create_from_options(options)
    return detector

if __name__ == "__main__":
    model_check()
    detector = model_load()

    cap = cv2.VideoCapture("../data/videos/input.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("cannot receive frame. Exit")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)    
        frame_timestamp_ms = int(1000 * frame_index / fps)
        face_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        if face_landmarker_result:
            h, w, _ = frame.shape
            for face in face_landmarker_result.face_landmarks:
                for lm in face:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)
        
        frame = cv2.resize(frame, (620, 960))
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()