# AIGT — Real-Time AI Traffic Management System

## 🚀 What it does
- Real-time vehicle detection & counting (YOLOv8)
- Traffic density estimation
- Emergency vehicle prioritization (Ambulance/Fire Brigade)
- Dynamic signal timer prediction (Custom PyTorch NN)

##      Computer Vision
- Detects traffic like cars,buses,etc count them and calculates the Traffic density
- we use yolov8 for this project with tensorRT in c++
- also done computer vision with python using cuda and done the same above things
- also done the segementation detection of traffic in both c++ and python
- 
## ⚡ Performance
- 50–70 FPS on NVIDIA RTX 3050 4GB GPU in c++ using TensorRT & CUDA
- 15-30 FPS on NVIDIA RTX 3050 4GB GPU in python using CUDA
- 10x speedup using TensorRT over standard inference

## Libraries Used
YOLOv8 | TensorRT | CUDA | OpenCV | PyTorch | C++ | Python

##      AI(ANN)
- created a NN and train it using some random datas
- this nn predicts the trafiic timer for dynamic timer system of the traffic system
- this takes input of lanes and provide the dynamic timer
- this nn also prioritise the emergency vehicles like ambulances and fire brigades
- this nn takes input as the traffic data from the computer vision and outputs/predicts the best dynamic timer per lane and priorities the emergency vehicles

## Libraries used
Python | Torch | Pandas | Numpy | SciKit-learn
