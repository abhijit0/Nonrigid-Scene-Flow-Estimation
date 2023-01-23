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
### Optical flow 
![image](https://user-images.githubusercontent.com/17523822/214089551-13898183-bc25-4247-80c8-1466d75396b9.png)
### Scene Flow 
![image](https://user-images.githubusercontent.com/17523822/214090702-c0d47948-efed-4b87-8792-df8a9ec0ed9e.png) 
|Optical Flow | Scene Flow |
|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
|Optical flow is a method to estimate the apparent motion of scene points from a sequence of images in 2d plane. | Scene Flow shows the threedimensional displacement vector between two frames in stereo setting|
|The flow vector is (u, v) representing flow in x and y directions respectively | Scene Flow between a pair of stereo images at time t and t+1 is given as (u, v, d0, Δd) where (u,v) is optical flow d0, Δd represent disparity info, capturing depth information |
| Challenges: Occlusions,Discontinuity in motion, Large motions | | 

# RAFT : Recurrent All-Pairs Field Transforms for Optical Flow
![image](https://user-images.githubusercontent.com/17523822/214092454-9343c1e9-3cc4-4f6e-bab5-ebcda5672a48.png)

- Deep Learning based optical flow estimation
- I/O : Images at t, t+1 ; O/P : Optical Flow b/w images t, t+1
- Reference Image t: Drives the iterative update module.
- Main Components :
  - Feature Extractor (CNN) – For generating feature maps of target images
  - Correlation Maps – For estimating flow at different resolutions
  - Iterative update operator – GRU unit which produces flow update at each step
