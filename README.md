# Deep Visual–Inertial Odometry for GPS-Denied UAV Navigation

This repository implements a **Deep Visual–Inertial Odometry (DeepVIO)** model for estimating UAV motion in **GPS-denied environments**. The system fuses monocular camera frames and IMU measurements using convolutional encoders, inertial temporal encoding, feature-level fusion, and an LSTM-based pose regressor.

The objective is to estimate the relative 6-DoF motion of a UAV:

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


where translation and rotation increments are predicted from image pairs and synchronized IMU windows.

---

## Motivation

GPS signals are often unavailable or unreliable in indoor, underground, urban-canyon, forest, disaster-response, and hostile environments. In these conditions, UAVs require onboard localization using local sensors.

The proposed DeepVIO model estimates motion using:

- **Monocular vision** for geometric and appearance-based motion cues.
- **IMU measurements** for high-rate inertial dynamics.
- **Neural fusion** for robust visual–inertial feature integration.
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

- \(\mathbf{p}_k \in \mathbb{R}^3\) is position,
- \(\mathbf{v}_k \in \mathbb{R}^3\) is velocity,
- \(\mathbf{R}_k \in SO(3)\) is attitude,
- \(\mathbf{b}_{a,k}\) is accelerometer bias,
- \(\mathbf{b}_{g,k}\) is gyroscope bias.

The IMU measurements are modeled as:

```math
\tilde{\boldsymbol{\omega}}_k =
\boldsymbol{\omega}_k + \mathbf{b}_{g,k} + \mathbf{n}_{g,k}
\]

\[
\tilde{\mathbf{a}}_k =
\mathbf{R}_k^\top
(
\mathbf{a}_k - \mathbf{g}
)
+
\mathbf{b}_{a,k}
+
\mathbf{n}_{a,k}
```

where \(\tilde{\boldsymbol{\omega}}_k\) and \(\tilde{\mathbf{a}}_k\) are gyroscope and accelerometer measurements, and \(\mathbf{n}_{g,k}\), \(\mathbf{n}_{a,k}\) are sensor noise terms.

Classical inertial propagation can be written as:

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
\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}
)
+
\mathbf{g}
\right)
\Delta t^2
```

\[
\mathbf{v}_{k+1}
=
\mathbf{v}_k
+
\left(
\mathbf{R}_k
(
\tilde{\mathbf{a}}_k - \mathbf{b}_{a,k}
)
+
\mathbf{g}
\right)
\Delta t
\]

\[
\mathbf{R}_{k+1}
=
\mathbf{R}_k
\exp
\left(
(
\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_{g,k})
\Delta t
\right)
\]

In this project, instead of explicitly solving the full state-estimation problem, a neural model learns the relative motion increment:

\[
\hat{\boldsymbol{\xi}}_k
=
f_{\theta}
(
\mathbf{I}_{k},
\mathbf{I}_{k+1},
\mathbf{u}_{k:k+m}
)
\]

where:

- \(\mathbf{I}_{k}\) and \(\mathbf{I}_{k+1}\) are consecutive monocular images,
- \(\mathbf{u}_{k:k+m}\) is the IMU window,
- \(f_{\theta}\) is the DeepVIO neural network,
- \(\hat{\boldsymbol{\xi}}_k \in \mathbb{R}^{6}\) is the predicted relative pose.

---

## Network Architecture

The proposed model contains four main modules:

### 1. Visual Encoder

Two consecutive RGB frames are concatenated into a 6-channel input:

\[
\mathbf{V}_k =
[
\mathbf{I}_{k},
\mathbf{I}_{k+1}
]
\]

A CNN extracts the visual motion feature:

\[
\mathbf{f}^{v}_k =
E_v(\mathbf{V}_k)
\]

### 2. Inertial Encoder

The IMU sequence is divided into short temporal windows:

\[
\mathbf{U}_k =
[
\mathbf{u}_k,
\mathbf{u}_{k+1},
...,
\mathbf{u}_{k+m}
]
\]

A 1D CNN extracts the inertial feature:

\[
\mathbf{f}^{i}_k =
E_i(\mathbf{U}_k)
\]

### 3. Visual–Inertial Fusion

The visual and inertial features are fused as:

\[
\mathbf{f}_k =
\mathcal{F}
(
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
)
\]

For concatenation-based fusion:

\[
\mathbf{f}_k =
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
\]

For gated soft fusion:

\[
\mathbf{f}_k =
\sigma
(
\mathbf{W}
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
)
\odot
[
\mathbf{f}^{v}_k,
\mathbf{f}^{i}_k
]
\]

where \(\sigma(\cdot)\) is the sigmoid function and \(\odot\) is element-wise multiplication.

### 4. Recurrent Pose Regressor

The fused features are passed to an LSTM:

\[
\mathbf{h}_k =
\mathrm{LSTM}
(
\mathbf{f}_k,
\mathbf{h}_{k-1}
)
\]

The relative pose is estimated by a regression head:

\[
\hat{\boldsymbol{\xi}}_k =
\mathbf{W}_p \mathbf{h}_k + \mathbf{b}_p
\]

The final predicted pose is:

\[
\hat{\boldsymbol{\xi}}_k =
[
\Delta x_k,
\Delta y_k,
\Delta z_k,
\Delta \phi_k,
\Delta \theta_k,
\Delta \psi_k
]
\]

---

## Policy-Based Visual Selection

The model includes a policy network that decides whether the current visual feature should be used or suppressed. This is useful when the camera is degraded by:

- motion blur,
- poor illumination,
- smoke,
- rain,
- fog,
- low texture,
- occlusion.

The decision is sampled using Gumbel-Softmax:

\[
\mathbf{d}_k =
\mathrm{GumbelSoftmax}
(
\pi_{\theta}
(
\mathbf{h}_{k-1},
\mathbf{f}^{i}_k
)
)
\]

The selected visual feature is:

\[
\tilde{\mathbf{f}}^{v}_k =
d_{k,0}\mathbf{f}^{v}_k
+
d_{k,1}\mathbf{0}
\]

Then the final fused feature becomes:

\[
\mathbf{f}_k =
\mathcal{F}
(
\tilde{\mathbf{f}}^{v}_k,
\mathbf{f}^{i}_k
)
\]

This mechanism allows the network to rely more heavily on IMU information when visual information is unreliable.

---

## GPS-Denied Navigation Method

The DeepVIO model can be used as the localization block of a GPS-denied UAV navigation pipeline.

At each time step:

1. The UAV captures a monocular image.
2. The IMU provides accelerometer and gyroscope measurements.
3. Consecutive images are paired.
4. IMU samples are synchronized into short windows.
5. DeepVIO predicts relative motion.
6. Relative poses are integrated into a local trajectory.
7. The local trajectory is used by the navigation controller.

The estimated trajectory is computed as:

\[
\hat{\mathbf{T}}_k =
\hat{\mathbf{T}}_{k-1}
\exp
(
\hat{\boldsymbol{\xi}}_k
)
\]

where \(\hat{\mathbf{T}}_k \in SE(3)\) is the estimated UAV pose.

This enables autonomous UAV navigation without GPS by using only onboard camera and IMU data.

---

## Model Input and Output

### Input

```python
img.shape = (batch_size, num_images, 3, height, width)
imu.shape = (batch_size, num_imu_samples, 6)
