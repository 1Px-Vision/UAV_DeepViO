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

## Model Input and Output

### Input

```python
img.shape = (batch_size, num_images, 3, height, width)
imu.shape = (batch_size, num_imu_samples, 6)
```

The six IMU channels are:

```text
ax, ay, az, gx, gy, gz
```

### Output

```python
poses.shape = (batch_size, num_images - 1, 6)
```

Each output pose is:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

---

## Google Colab Trajectory Animation

The following code creates a simple animation of the estimated UAV trajectory from the DeepVIO relative pose output.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def integrate_relative_poses(relative_poses):
    """
    Integrates relative DeepVIO pose increments into a 2D trajectory.

    relative_poses:
        numpy array with shape (N, 6)
        each row = [dx, dy, dz, droll, dpitch, dyaw]
    """

    x, y, z = 0.0, 0.0, 0.0
    yaw = 0.0

    trajectory = []

    for pose in relative_poses:
        dx, dy, dz, droll, dpitch, dyaw = pose

        yaw += dyaw

        global_dx = np.cos(yaw) * dx - np.sin(yaw) * dy
        global_dy = np.sin(yaw) * dx + np.cos(yaw) * dy

        x += global_dx
        y += global_dy
        z += dz

        trajectory.append([x, y, z])

    return np.array(trajectory)


def animate_vio_trajectory(trajectory):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title("Deep Visual-Inertial Odometry for GPS-Denied UAV Navigation")
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.grid(True)

    margin = 1.0
    ax.set_xlim(trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin)
    ax.set_ylim(trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin)

    path_line, = ax.plot([], [], linewidth=2, label="Estimated VIO trajectory")
    drone_point, = ax.plot([], [], marker="o", markersize=8, label="UAV")
    ax.legend()

    def init():
        path_line.set_data([], [])
        drone_point.set_data([], [])
        return path_line, drone_point

    def update(frame):
        path_line.set_data(
            trajectory[:frame + 1, 0],
            trajectory[:frame + 1, 1]
        )
        drone_point.set_data(
            [trajectory[frame, 0]],
            [trajectory[frame, 1]]
        )
        return path_line, drone_point

    animation = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        init_func=init,
        interval=120,
        blit=True
    )

    plt.close(fig)
    return animation


# Example:
# poses is the model output with shape (batch_size, sequence_length, 6)

relative_poses = poses[0].detach().cpu().numpy()
trajectory = integrate_relative_poses(relative_poses)

animation = animate_vio_trajectory(trajectory)

HTML(animation.to_jshtml())
```

---

## Save Animation as GIF

```python
animation.save("deep_vio_gps_denied.gif", writer="pillow", fps=10)
```

Then include the GIF in the README:

```markdown
## GPS-Denied VIO Animation

![DeepVIO GPS-denied trajectory](assets/deep_vio_gps_denied.gif)
```

---

## Save Animation as MP4

```python
!apt-get install -y ffmpeg
animation.save("deep_vio_gps_denied.mp4", writer="ffmpeg", fps=10)
```

Then include the MP4 as a clickable link:

```markdown
## GPS-Denied VIO Video

[Watch the DeepVIO GPS-denied trajectory animation](assets/deep_vio_gps_denied.mp4)
```

For better GitHub compatibility, use a GIF preview linked to the MP4:

```markdown
[![DeepVIO GPS-denied trajectory](assets/deep_vio_gps_denied.gif)](assets/deep_vio_gps_denied.mp4)
```

---

## Application Scenarios

The proposed DeepVIO framework is useful for UAV navigation in:

- indoor environments,
- tunnels and mines,
- urban canyons,
- disaster-response scenarios,
- forest inspection,
- GPS-jammed environments,
- search-and-rescue missions,
- low-altitude autonomous flight.

---

## Method Summary

The proposed DeepVIO model estimates UAV motion by learning a direct mapping from monocular image pairs and IMU windows to relative 6-DoF pose increments. The visual encoder captures frame-to-frame motion cues, while the inertial encoder extracts short-term dynamics from accelerometer and gyroscope data. A fusion module combines both modalities, and an LSTM models temporal dependencies across the flight sequence.

A policy network improves robustness by suppressing unreliable visual features under degraded image conditions. This architecture provides an onboard localization method for GPS-denied UAV navigation, allowing a UAV to estimate its trajectory using only local visual and inertial sensing.

---

## Repository Description

Deep Visual-Inertial Odometry model for GPS-denied UAV navigation using monocular image pairs, IMU windows, neural feature fusion, LSTM pose regression, policy-based visual selection, and Google Colab trajectory animation.
