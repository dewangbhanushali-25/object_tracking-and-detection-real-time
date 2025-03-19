#YOLOv8 + DeepSORT Object Detection and Tracking
This repository implements a robust object detection and tracking system combining YOLOv8 for detection, DeepSORT for multi-object tracking, and motion estimation for improved performance.
Overview
The system uses a three-stage approach:

Object Detection: YOLOv8 identifies objects in individual frames
Motion Estimation: Predicts object movement between frames
Object Tracking: DeepSORT (Deep Simple Online and Realtime Tracking) maintains object identities across frames

This combination creates a powerful solution for applications like surveillance, traffic monitoring, sports analytics, and more.
Features

Real-time object detection using state-of-the-art YOLOv8
Motion estimation to predict object positions between frames
Persistent object tracking across video frames with DeepSORT
Convolutional Neural Network (CNN) based appearance feature extraction
Support for multiple object classes
Configurable detection confidence thresholds
Performance optimizations for real-time processing
