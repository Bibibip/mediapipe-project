# Mediapipe_Project
mediapipe를 사용하여 python으로 이미지, 동영상, 웹캡에서 face landmark를 추출하고 시각화하는 것을 목표로 하였다.

## 개발 환경
- **python 버전**: (현재) 3.10
- **Libraries**: OpenCV
- **Framework**: MediaPipe

## 실행 방법
### 1. 가상환경 생성 및 실행

다른 라이브러리와의 충돌을 피하기 위해 독립적인 가상환경을 구축한다.

- 가상환경 생성:
  ```bash
  python -m venv (가상환경 이름)
- 가상환경 활성화:
  ```bash
  .\(가상환경 이름)\Scripts\activate
 
### 2. opencv & mediapipe 라이브러리 설치

가상환경을 활성화한 후, 구현에 필요한 opencv와 mediapipe를 설치해야 한다.
```bash
pip install opencv-python mediapipe
```

### 3. 모델 저장 및 설정

face_landmarker.task 모델 파일은 [공식 문서](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index?hl=ko#models)의 링크를 통해 다운로드 받을 수 있다.

모델이 저장된 경로를 BaseOptions의 매개변수를 통해 지정하므로 모델이 저장된 경로를 잘 기억한다. (깃허브에 올린 코드의 경우, url을 통해 모델이 로컬에 없는 경우 자동으로 다운로드 된다.)

예시)
```bash
model_path = '/absolute/path/to/face_landmarker.task'
base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.IMAGE)
```

### 4. 코드 요약

**1. 이미지**

적용할 이미지를 불러오고 랜드마크를 추출한다. 
```bash
img = cv2.imread("이미지 경로")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format = mp.ImageFormat.SRGB, data = img_rgb
)
```

FaceLandmarker 실행을 위해 모델에 이미지를 매개변수로 전달한다. 
```bash
result = detector.detect(mp_image)
```

**2. 동영상**

동영상의 경우, running_mode를 VIDEO로 변경한다.

```bash
running_mode = VisionRunningMode.VIDEO
```

유튜브에서 동영상을 다운받아 사용하는 경우, 터미널에 아래와 같이 입력한다.

```bash
pip install yt-dlp
```
```bash
yt-dlp "유튜브 동영상 링크"
```

**3. 웹캠**

웹캠의 경우, 동영상에 적용할 때와 매우 유사하나 영상을 따로 지정하지 않고 `cap = cv2.VideoCapture(0)`를 통해 랜드마크 추출을 적용할 프레임을 지정한다.


### 5. 코드 실행

```bash
python 파일명.py
```

