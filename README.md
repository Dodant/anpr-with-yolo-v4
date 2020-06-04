# ANPR-with-Yolo-v4
Paper Yolo v4: https://arxiv.org/abs/2004.10934

About Darknet : http://pjreddie.com/darknet/

## Classes
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


## Example
### Original Image
![testfile](https://user-images.githubusercontent.com/20153952/83719439-09da3780-a672-11ea-9143-2ce1d9f921f1.jpg)
### Prediction Image
`./darknet detector test data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/testfile.jpg`

![predictions](https://user-images.githubusercontent.com/20153952/83719443-0e9eeb80-a672-11ea-8771-761a175f48e6.jpg)



## References
- 
- Keras-yolov3
  - weights to h5 - https://github.com/qqwweee/keras-yolo3/blob/master/convert.py
- Core ML
  - Mish - https://qiita.com/TakaoNarikawa/items/e4521fd8c7a522e9d4fd
  - https://gist.github.com/TakaoNarikawa
  - https://github.com/Ma-Dan/YOLOv3-CoreML
