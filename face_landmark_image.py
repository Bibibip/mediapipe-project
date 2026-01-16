import cv2
import os
from urllib.request import urlretrieve
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision   # mediapipe tasks 비전 라이브러리(인코더-디코더)

dir_path = "landmark_model"
model_name = "face_landmarker.task"
model_path = os.path.join(dir_path, model_name)
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def model_check():
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if not os.path.exists(model_path):
        urlretrieve(model_url, model_path)
        print("face mesh file download")
    
    print(f"model is loaded : {model_path}")

def model_load():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = model_path),
        running_mode = VisionRunningMode.IMAGE,
        num_faces=1
    )

    detector = FaceLandmarker.create_from_options(options)
    return detector

if __name__ == "__main__":
    model_check()
    detector = model_load()

    # 이미지 불러와서 랜드마크 추출
    img = cv2.imread("../data/images/myFace.jpg")
    if img is None:
        print("이미지 로드 실패")
        exit()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = img_rgb
    )

    result = detector.detect(mp_image)

    # 랜드마크 좌표 그리기
    h, w, _ = img.shape

    if result.face_landmarks:
        for face in result.face_landmarks:
            for landmark in face:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img, (x,y), 2, (0,255,0), -1)
    
    else:
        print("얼굴 검출 실패")

    cv2.imshow("Face Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()