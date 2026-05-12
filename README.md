# Deep Visual–Inertial Odometry for GPS-Denied UAV Navigation

This repository implements a **Deep Visual–Inertial Odometry (DeepVIO)** model for estimating UAV motion in **GPS-denied environments**. The system fuses monocular camera frames and inertial measurement unit (IMU) data using convolutional encoders, temporal inertial encoding, feature-level fusion, and an LSTM-based pose regressor.

The objective is to estimate the relative 6-DoF motion of a UAV:

```math
\hat{\boldsymbol{\xi}}_k =
[
\Delta x_k,
\Delta y_k,
\Delta z_k,
\Delta \phi_k,
\Delta \theta_k,
\Delta \psi_k
]
```

where translation and rotation increments are predicted from monocular image pairs and synchronized IMU windows.

---

## Motivation

GPS signals are often unavailable or unreliable in indoor, underground, urban-canyon, forest, disaster-response, and hostile environments. In these conditions, UAVs require onboard localization using local sensors.

The proposed DeepVIO model estimates motion using:

- **Monocular vision** for geometric and appearance-based motion cues.
- **IMU measurements** for high-rate inertial dynamics.
- **Neural feature fusion** for robust visual–inertial integration.
- **Recurrent temporal modeling** for smooth trajectory estimation.
- **Policy-based visual selection** to reduce the effect of degraded visual frames.

This makes the method suitable for **GPS-denied autonomous UAV navigation**.

---

## Visual–Inertial Odometry Formulation

At time step \(k\), the UAV state can be represented as:

```math
\mathbf{x}_k =
[
\mathbf{p}_k,
\mathbf{v}_k,
\mathbf{R}_k,
\mathbf{b}_{a,k},
\mathbf{b}_{g,k}
]
```

where:

- \(\mathbf{p}_k \in \mathbb{R}^3\) is the UAV position,
- \(\mathbf{v}_k \in \mathbb{R}^3\) is the UAV velocity,
- \(\mathbf{R}_k \in SO(3)\) is the UAV attitude matrix,
- \(\mathbf{b}_{a,k}\) is the accelerometer bias,
- \(\mathbf{b}_{g,k}\) is the gyroscope bias.

The gyroscope measurement model is:

```math
\tilde{\boldsymbol{\omega}}_k =
\boldsymbol{\omega}_k
+
\mathbf{b}_{g,k}
+
\mathbf{n}_{g,k}
```

The accelerometer measurement model is:

```math
\tilde{\mathbf{a}}_k =
\mathbf{R}_k^\top
(
\mathbf{a}_k
-
\mathbf{g}
)
+
\mathbf{b}_{a,k}
+
\mathbf{n}_{a,k}
```

where \(\tilde{\boldsymbol{\omega}}_k\) and \(\tilde{\mathbf{a}}_k\) are the measured angular velocity and acceleration, while \(\mathbf{n}_{g,k}\) and \(\mathbf{n}_{a,k}\) are sensor noise terms.

The classical inertial position propagation is:

```math
\mathbf{p}_{k+1}
=
\mathbf{p}_k
+
\mathbf{v}_k \Delta t
+
\frac{1}{2}
\left(
\mathbf{R}_k
(
\tilde{\mathbf{a}}_k
-
\mathbf{b}_{a,k}
)
+
\mathbf{g}
\right)
\Delta t^2
```

The velocity propagation is:

```math
\mathbf{v}_{k+1}
=
\mathbf{v}_k
+
\left(
\mathbf{R}_k
(
\tilde{\mathbf{a}}_k
-
\mathbf{b}_{a,k}
)
+
\mathbf{g}
\right)
\Delta t
```

The attitude propagation is:

```math
\mathbf{R}_{k+1}
=
\mathbf{R}_k
\exp
\left(
(
\tilde{\boldsymbol{\omega}}_k
-
\mathbf{b}_{g,k}
)
\Delta t
\right)
```

Instead of explicitly solving the full visual–inertial state-estimation problem, the proposed DeepVIO model learns the relative motion increment directly:

```math
\hat{\boldsymbol{\xi}}_k
=
f_{\theta}
(
\mathbf{I}_{k},
\mathbf{I}_{k+1},
\mathbf{u}_{k:k+m}
)
```

where:

- \(\mathbf{I}_{k}\) and \(\mathbf{I}_{k+1}\) are consecutive monocular images,
- \(\mathbf{u}_{k:k+m}\) is the synchronized IMU window,
- \(f_{\theta}\) is the DeepVIO neural network,
- \(\hat{\boldsymbol{\xi}}_k \in \mathbb{R}^{6}\) is the predicted relative pose.

The predicted relative pose is:

```math
\hat{\boldsymbol{\xi}}_k =
[
\Delta x_k,
\Delta y_k,
\Delta z_k,
\Delta \phi_k,
\Delta \theta_k,
\Delta \psi_k
]
```

---

## Network Architecture

The DeepVIO model contains four main modules:

1. Visual encoder  
2. Inertial encoder  
3. Visual–inertial fusion module  
4. Recurrent pose regressor  

---

## 1. Visual Encoder

Two consecutive RGB frames are concatenated into a 6-channel input:

```math
\mathbf{V}_k =
[
\mathbf{I}_{k},
\mathbf{I}_{k+1}
]
```

A convolutional neural network extracts the visual motion feature:

```math
\mathbf{f}^{v}_k =
E_v
(
\mathbf{V}_k
)
```

where \(E_v(\cdot)\) is the visual encoder.

---

## 2. Inertial Encoder

The IMU sequence is divided into short temporal windows:

```math
\mathbf{U}_k =
[
\mathbf{u}_k,
\mathbf{u}_{k+1},
...,
\mathbf{u}_{k+m}
]
```

Each IMU sample contains accelerometer and gyroscope measurements:

```math
\mathbf{u}_k =
[
a_x,
a_y,
a_z,
\omega_x,
\omega_y,
\omega_z
]
```

A 1D convolutional encoder extracts the inertial feature:

```math
\mathbf{f}^{i}_k =
E_i
(
\mathbf{U}_k
)
```

where \(E_i(\cdot)\) is the inertial encoder.

---

## 3. Visual–Inertial Fusion

The visual and inertial features are fused as:

```math
\mathbf{f}_k =
\mathcal{F}
(
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
)
```

For direct concatenation:

```math
\mathbf{f}_k =
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
```

For soft gated fusion:

```math
\mathbf{f}_k =
\sigma
\left(
\mathbf{W}
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
+
\mathbf{b}
\right)
\odot
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
```

where:

- \(\sigma(\cdot)\) is the sigmoid activation,
- \(\odot\) is element-wise multiplication,
- \(\mathbf{W}\) and \(\mathbf{b}\) are learnable fusion parameters.

---

## 4. Recurrent Pose Regressor

The fused feature is passed through an LSTM to model temporal motion dependencies:

```math
\mathbf{h}_k =
\mathrm{LSTM}
(
\mathbf{f}_k,
\mathbf{h}_{k-1}
)
```

The relative pose is estimated using a regression head:

```math
\hat{\boldsymbol{\xi}}_k =
\mathbf{W}_p
\mathbf{h}_k
+
\mathbf{b}_p
```

The final predicted pose increment is:

```math
\hat{\boldsymbol{\xi}}_k =
[
\Delta x_k,
\Delta y_k,
\Delta z_k,
\Delta \phi_k,
\Delta \theta_k,
\Delta \psi_k
]
```

---

## Policy-Based Visual Selection

The model includes a policy network that decides whether the current visual feature should be used or suppressed. This is useful when the camera is affected by:

- motion blur,
- poor illumination,
- smoke,
- rain,
- fog,
- low texture,
- occlusion.

The decision is sampled using Gumbel-Softmax:

```math
\mathbf{d}_k =
\mathrm{GumbelSoftmax}
\left(
\pi_{\theta}
(
\mathbf{h}_{k-1},
\mathbf{f}^{i}_k
)
\right)
```

The selected visual feature is:

```math
\tilde{\mathbf{f}}^{v}_k =
d_{k,0}
\mathbf{f}^{v}_k
+
d_{k,1}
\mathbf{0}
```

The final fused feature becomes:

```math
\mathbf{f}_k =
\mathcal{F}
(
\tilde{\mathbf{f}}^{v}_k,
\mathbf{f}^{i}_k
)
```

This mechanism allows the network to rely more heavily on IMU information when visual information is unreliable.

---

## GPS-Denied Navigation Method

The DeepVIO model can be used as the localization block of a GPS-denied UAV navigation pipeline.

![](https://github.com/1Px-Vision/UAV_DeepViO/blob/main/PF_Deep_Prior.jpg)

At each time step:

1. The UAV captures a monocular image.
2. The IMU provides accelerometer and gyroscope measurements.
3. Consecutive images are paired.
4. IMU samples are synchronized into short temporal windows.
5. DeepVIO predicts the relative 6-DoF motion.
6. Relative poses are integrated into a local trajectory.
7. The estimated trajectory is used by the navigation controller.

The global UAV pose is updated by integrating the predicted relative motion:

```math
\hat{\mathbf{T}}_k =
\hat{\mathbf{T}}_{k-1}
\exp
(
\hat{\boldsymbol{\xi}}_k
)
```

where \(\hat{\mathbf{T}}_k \in SE(3)\) is the estimated UAV pose.

The estimated trajectory is:

```math
\hat{\mathcal{T}}
=
\{
\hat{\mathbf{T}}_0,
\hat{\mathbf{T}}_1,
...,
\hat{\mathbf{T}}_N
\}
```

This enables autonomous UAV navigation without GPS by using only onboard visual and inertial sensing.

---

## Particle Filter Monocular VO + CNN Deep Motion Prior

A Python-based **GPS-denied monocular visual odometry navigation system** that combines classical feature-based VO, a **Particle Filter pose estimator**, and a lightweight **CNN Deep Motion Prior** for trajectory smoothing and motion regularization.

### Demo Video

The following video shows the output of the **Particle Filter Monocular VO + CNN Deep Motion Prior** system, including the marked navigation video, estimated trajectory, and visual odometry mapping.

[](https://github.com/1Px-Vision/UAV_DeepViO/blob/main/converted_V2.mp4)

[⬇ Download raw MP4](https://github.com/1Px-Vision/UAV_DeepViO/raw/main/converted_V2.mp4)

The system reads video frames from a monocular camera sequence, estimates frame-to-frame motion, filters noisy pose updates using a particle filter, learns a short-horizon motion prior with a CNN, and generates trajectory plots, evaluation metrics, and a marked navigation video.

The main idea is to combine:

1. **Feature-based monocular VO**
   - ORB/KLT feature tracking
   - Essential matrix estimation
   - Relative pose recovery
   - Incremental trajectory reconstruction

2. **Particle Filter localization**
   - Maintains multiple pose hypotheses
   - Reduces noisy VO updates
   - Improves robustness under drift, feature loss, and scale uncertainty

3. **CNN Deep Motion Prior**
   - Learns motion consistency from recent visual features
   - Regularizes translation and heading updates
   - Improves the smoothness of the estimated trajectory

4. **Evaluation and visualization**
   - RMSE and error calculation when ground truth is available
   - Aligned trajectory plots
   - Marked output video with navigation map and motion indicators

---

## Main Features

- Monocular visual odometry from video input
- Particle Filter-based trajectory refinement
- CNN Deep Motion Prior for learned motion smoothing
- Optional GPU acceleration with PyTorch
- Automatic fallback if PyTorch is unavailable
- KITTI-style calibration file support
- Ground truth trajectory evaluation
- RMSE, Absolute Trajectory Error, and frame-wise error calculation
- Output video with marked VO navigation overlay
- CSV, TUM, and KITTI trajectory export
- 2D trajectory visualization
- Headless-compatible execution for Linux, servers, and Google Colab

---

## Project Structure

```text
Particle-Filter-Monocular-VO-CNN-Prior/
│
├── mono_vo_cnn_deep_prior_particle_gpu_improved_prior.py
├── README.md
│
├── kitti06/
│   ├── video.mp4
│   ├── 06.txt
│   ├── groundtruth.txt
│   └── times.txt
│
└── vo_output_particle_cnn/
    ├── trajectory.csv
    ├── trajectory_tum.txt
    ├── trajectory_kitti.txt
    ├── trajectory_xz.png
    ├── trajectory_aligned_xz.png
    ├── marked_vo_navigation_video.mp4
    ├── evaluation.txt
    └── cnn_motion_prior.pt
```
### Urban example navigation

![](https://github.com/1Px-Vision/UAV_DeepViO/blob/main/GPS_denied_urban.jpg)
```
python mono_vo_cnn_deep_prior_particle_gpu_improved_prior.py \
    --data_dir kitti06 \
    --outdir vo_output_improved_deep_prior \
    --scale_mode auto \
    --use_cnn_prior \
    --prior_blend 0.25 \
    --prior_train_steps 2 \
    --prior_height 160 \
    --prior_width 480 \
    --motion_smooth 0.20 \
    --particle_count 1200 \
    --particle_process_noise 0.030 \
    --particle_measurement_noise 0.25 \
    --gpu
```
---
### Forest navigation

![](https://github.com/1Px-Vision/UAV_DeepViO/blob/main/GPS_denied_forest.jpg)

<p align="center">
  <a href="https://www.kapwing.com/e/6a0338771803f40fa7cb494a">
    ▶ VIO + Geo-Location GPS-Denied Drone Navigation Demo
  </a>
</p>

![](https://github.com/1Px-Vision/UAV_DeepViO/blob/main/GPS_denied_forest_2.jpg)

<p align="center">
  <b>VIO + Geo-Location GPS-Denied Forest Navigation</b><br>
  Click demo video.
</p>

<p align="center">
  <a href="https://www.kapwing.com/e/6a033f1f917c991a7529dd9d">▶ VIO + Geo-Location GPS-Denied Forest Navigation</a>
</p>


## Method Summary

The proposed DeepVIO model estimates UAV motion by learning a direct mapping from monocular image pairs and IMU windows to relative 6-DoF pose increments. The visual encoder captures frame-to-frame motion cues, while the inertial encoder extracts short-term dynamics from accelerometer and gyroscope data. A fusion module combines both modalities, and an deep-prior models temporal dependencies across the flight sequence.

A policy network improves robustness by suppressing unreliable visual features under degraded image conditions. This architecture provides an onboard localization method for GPS-denied UAV navigation, allowing a UAV to estimate its trajectory using only local visual and inertial sensing.

---

## Repository Description

Deep Visual-Inertial Odometry model for GPS-denied UAV navigation using monocular image pairs, IMU windows, neural feature fusion, deep-prior pose regression, policy-based visual selection, and Google Colab trajectory animation.
