# ANPR-with-Yolo-v4
**ANPR (Automatic Number-Plate Recognition)** : 차량번호판 자동 인식 프로그램

**Yolo (You Only Look Once)** : One-Stage Object Detector

About Darknet : http://pjreddie.com/darknet/

## Download Model
- `.weights` : https://drive.google.com/file/d/1b3rYgP48z_NGvSuNoMKDvxXzYmray_Qr/view?usp=sharing
- `.mlmodel` : https://drive.google.com/file/d/1eREdAVoiOVlAiPOxv5c_Sc5EBPybOIUq/view?usp=sharing

## Classes
- car
- license_plate

## Training 
Labeling Tool : https://github.com/AlexeyAB/Yolo_mark

Darknet (Yolov4) : https://github.com/AlexeyAB/darknet

|Cloud Service|GPU|Traing Data|훈련 횟수|시간|
|---|---|:---:|:---:|:---:|
|GCP(Google Cloud Platform)|Nvidia Tesla P100|2600여장|4000회|5h|

`./darknet detector train data/obj.data cfg/yolov4_ANPR.cfg yolov4.conv.137 -gpu 0`

## Usage (test)
1. `git clone https://github.com/AlexeyAB/darknet`
2. `cd darknet`
3. 사용하는 환경에 맞게 Makefile 설정  `vi Makefile`
```
GPU=0		# GPU 사용 시 1로 변경
CUDNN=0		# cuDNN 사용 시 1로 변경 (NVIDIA)
CUDNN_HALF=0
OPENCV=0	# OpenCV 사용 시 1로 변경
AVX=0
OPENMP=0
LIBSO=0

...
...
```
4. `make`
- 기본 패키지 : make, gcc, pkg-config (없다면 `sudo apt-get install …`로 설치)
  
5. `data/*`, `cfg/yolov4-ANPR.cfg`, ` backup/yolov4-ANPR.weights` 다운로드 

### image

`./darknet detector test data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/(이미지파일.jpg)`

> 반드시 `.jpg` 이미지 사용 
### video
`./darknet detector demo data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/(동영상파일.mp4)`

### webcam
`./darknet detector demo data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights`


## Example
### Prediction Image
`./darknet detector test data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/testfile.jpg`
```
Loading weights from backup/yolov4-ANPR.weights...
 seen 64, trained: 256 K-images (4 Kilo-batches_64)
Done! Loaded 162 layers from weights-file
data/testfile.jpg: Predicted in 9325.005000 milli-seconds.
car: 63%
car: 98%
license_plate: 96%
car: 47%
car: 61%
car: 30%
```
![predictions](https://user-images.githubusercontent.com/20153952/83719443-0e9eeb80-a672-11ea-8771-761a175f48e6.jpg)



## References
- Paper
  - [You Only Look Once : Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
  - [YOLO9000 : Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
  - [YOLOv3 : An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  - [YOLOv4：Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)
- Keras-yolov3
  - weights to h5 : https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
  - weights to mlmodel : https://gist.github.com/TakaoNarikawa/aef13571eec97d78603688eef05b5389
  - Mish : https://qiita.com/TakaoNarikawa/items/e4521fd8c7a522e9d4fd
- Core ML
  - https://gist.github.com/TakaoNarikawa
  - https://github.com/Ma-Dan/YOLOv3-CoreML

