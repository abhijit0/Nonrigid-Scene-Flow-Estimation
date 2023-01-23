# Nonrigid-Scene-Flow-Estimation

## Project Overview:
- Motivation: RAFT is State of the art for optical flow estimation. Why not
apply for scene flow ?

- Goal : Scene Flow Estimation using Raft by training on Flyingthigs3d dataset
and later fine tuning on Kitti to evaluate against Kitti scene flow benchmark

- Scene Flow is more robust representation of change in scene as it also
incorporates depth information where optical flow does not.

- Non rigid scene flow because we cannot model all the motion as rigidly
moving objects e.g human movements.

## Optical Flow v/s Scene Flow 
|Optical Flow | Scene Flow |
|--------------------------------------------------------------------------------------------------------------|---------------------------------------|
|Optical flow is a method to estimate the apparent motion of scene points from a sequence of images in 2d plane. | Scene Flow shows the threedimensional displacement vector between two frames in stereo setting|

|![image](https://user-images.githubusercontent.com/17523822/214089551-13898183-bc25-4247-80c8-1466d75396b9.png)| 
![image](https://user-images.githubusercontent.com/17523822/214090702-c0d47948-efed-4b87-8792-df8a9ec0ed9e.png) |
