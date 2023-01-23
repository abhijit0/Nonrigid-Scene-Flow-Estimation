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
|---|---|
![image](https://user-images.githubusercontent.com/17523822/214089551-13898183-bc25-4247-80c8-1466d75396b9.png)
