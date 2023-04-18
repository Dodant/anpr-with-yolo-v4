# ALPR-with-Yolo-v4
ALPR with YOLOv4 is an advanced Automatic License Plate Recognition (ALPR) system that leverages the powerful YOLOv4 (You Only Look Once) one-stage object detection framework. It can efficiently and accurately detect and recognize vehicle license plates in real-time.

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

|Cloud Service|GPU|Traing Data|Training Iterations|Time|
|---|---|:---:|:---:|:---:|
|GCP(Google Cloud Platform)|Nvidia Tesla P100|Over 2600 images|4000 iterations|5h|

`./darknet detector train data/obj.data cfg/yolov4_ANPR.cfg yolov4.conv.137 -gpu 0`

## Usage (test)
1. `git clone https://github.com/AlexeyAB/darknet`
2. `cd darknet`
3. Configure Makefile according to your environment:  `vi Makefile`
```
GPU=0        # Change to 1 if using GPU
CUDNN=0      # Change to 1 if using cuDNN (NVIDIA)
CUDNN_HALF=0
OPENCV=0     # Change to 1 if using OpenCV
AVX=0
OPENMP=0
LIBSO=1      # Generate libdarknet.so

...
...
```
4. `make`
- Required packages: make, gcc, pkg-config (if not installed, use `sudo apt-get install …` to install)
  
5. Download `data/*`, `cfg/yolov4-ANPR.cfg`, and `backup/yolov4-ANPR.weights` 

### image

`./darknet detector test data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/(이미지파일.jpg)`

> Make sure to use `.jpg` images 
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


### Prediction Video
- `./darknet detector demo data/obj.data cfg/yolov4-ANPR.cfg backup/yolov4-ANPR.weights data/testvideo.jpg`
- `python darknet_video.py`

Demo Video Link (1) : https://drive.google.com/file/d/1DGmF2bwtDMe1y-wNuv_YT827Vr6Y8Q2m/view?usp=sharing

Demo Video Link (2) : https://drive.google.com/file/d/1nJjIQFcrYRYSJ0n9FK0-x_Fk6HrULsZY/view?usp=sharing

## References
- Papers
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


## Presentation
- 발표자료 : https://drive.google.com/file/d/1yhhIZ0ZU5MIZZar-WTBGgEHwLkOHcewe/view?usp=sharing

- 발표영상 : https://www.youtube.com/watch?v=H3-SVf0Ps4c

- Capstone Design : https://github.com/kwanghoon/CapstoneDesign
