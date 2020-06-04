# ANPR-with-Yolo-v4

## Class
- car
- license_plate

## Training 
Labeling Tool : https://github.com/AlexeyAB/Yolo_mark

Darknet (Yolov4) : https://github.com/AlexeyAB/darknet

|Cloud Service|GPU|Traing Data|훈련 횟수|시간|
|---|---|:---:|:---:|:---:|
|GCP(Google Cloud Platform)|Nvidia Tesla P100|2600여장|4000회|5h|

## Usage (test)
### image
`./darknet detector test data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/(이미지파일.jpg)`

> 반드시 `.jpg` 이미지 사용 
### video
`./darknet detector demo data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/(동영상파일.mp4)`

### webcam
`./darknet detector demo data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights`
