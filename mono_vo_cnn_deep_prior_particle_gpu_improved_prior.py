#!/usr/bin/env python3
"""
Monocular Visual Odometry + CNN Deep Motion Prior + Particle Filter + GPU Acceleration
with aligned x-z trajectory map, live RMSE/error values, and an error curve in the output video.

Expected directory:

kitti06/
    06.txt
    groundtruth.txt
    times.txt
    video.mp4

Outputs:

vo_output_cnn_prior_particle_gpu/
    trajectory.csv
    trajectory_tum.txt
    trajectory_kitti.txt
    trajectory_xz.png
    trajectory_aligned_xz.png
    error_over_time.png
    trajectory_error.csv
    marked_vo_video.mp4
    evaluation.txt
    cnn_motion_prior.pt

Main updates:
    1. Video trajectory map uses display-only Umeyama alignment when GT exists.
    2. Video trajectory map is stable, centered, and grid-free.
    3. Saved matplotlib trajectory plots are grid-free.
    4. Raw trajectory.csv and pose outputs remain unchanged.
    5. Improved residual-attention CNN Deep Motion Prior is used for online scale regularization.
    6. 3D bootstrap Particle Filter is added to reduce trajectory jitter and RMSE.
    7. Optional GPU acceleration is added for OpenCV preprocessing/ORB and PyTorch CNN prior.
    8. RMSE, MAE, max error, final drift, and per-frame position errors are computed.
    9. Live online ATE RMSE, x-z RMSE, current error, MAE, max error, and a mini error curve
       are displayed directly on the trajectory map during navigation.
    10. The deep prior now uses residual blocks, SE attention, a frame-difference input channel,
        log-scale training, online augmentation, EMA stabilization, and uncertainty-aware confidence.

Run:

    python mono_vo_cnn_deep_prior_particle_gpu_rmse_map.py \
        --data_dir kitti06 \
        --outdir vo_output_cnn_prior_particle_gpu \
        --scale_mode auto \
        --use_cnn_prior \
        --prior_blend 0.25 \
        --motion_smooth 0.20 \
        --particle_count 1000 \
        --particle_process_noise 0.035 \
        --particle_measurement_noise 0.28 \
        --gpu

Important:
    Monocular VO does not recover absolute scale by itself.
    If groundtruth.txt is available, --scale_mode auto uses GT frame-to-frame
    displacement as metric scale. The video map alignment is display-only and
    does not modify trajectory.csv. The Particle Filter is applied to the
    estimated VO position before writing the final trajectory.
"""

import argparse
import csv
import math
import random
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CUDA_CV_AVAILABLE = False
CUDA_ORB_AVAILABLE = False

try:
    CUDA_CV_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
    CUDA_ORB_AVAILABLE = CUDA_CV_AVAILABLE and hasattr(cv2, "cuda_ORB")
except Exception:
    CUDA_CV_AVAILABLE = False
    CUDA_ORB_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


# ============================================================
# File utilities
# ============================================================

def read_numbers_from_line(line: str) -> List[float]:
    return [
        float(x)
        for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    ]


def find_dataset_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    files = list(data_dir.iterdir()) if data_dir.exists() else []
    videos = sorted([p for p in files if p.suffix.lower() in VIDEO_EXTS])

    calib = None
    for name in ["06.txt", "calib.txt", "calibration.txt"]:
        p = data_dir / name
        if p.exists():
            calib = p
            break

    gt = None
    for name in ["groundtruth.txt", "gt.txt", "poses.txt"]:
        p = data_dir / name
        if p.exists():
            gt = p
            break

    times = None
    for name in ["times.txt", "timestamps.txt"]:
        p = data_dir / name
        if p.exists():
            times = p
            break

    return {
        "video": videos[0] if videos else None,
        "calib": calib,
        "groundtruth": gt,
        "times": times,
    }


def read_times(times_file: Optional[Path], num_frames: int, fps: float) -> List[float]:
    if times_file is not None and times_file.exists():
        values = []
        with open(times_file, "r", encoding="utf-8") as f:
            for line in f:
                nums = read_numbers_from_line(line)
                if nums:
                    values.append(float(nums[0]))

        if len(values) >= num_frames:
            return values[:num_frames]

        if len(values) > 0:
            dt = float(np.median(np.diff(values))) if len(values) >= 2 else 1.0 / max(fps, 1.0)

            while len(values) < num_frames:
                values.append(values[-1] + dt)

            return values

    dt = 1.0 / max(fps, 1.0)
    return [i * dt for i in range(num_frames)]


# ============================================================
# Calibration and ground truth
# ============================================================

def read_kitti_calibration(calib_file: Optional[Path]) -> np.ndarray:
    K_default = np.array(
        [
            [718.8560, 0.0, 607.1928],
            [0.0, 718.8560, 185.2157],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if calib_file is None or not calib_file.exists():
        print("[WARN] Calibration file not found. Using KITTI default K.")
        return K_default

    projections = {}

    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            nums = read_numbers_from_line(value)

            if len(nums) >= 12:
                projections[key] = np.array(nums[:12], dtype=np.float64).reshape(3, 4)

            elif len(nums) >= 9 and key.upper() in {"K", "K0", "CAMERA_MATRIX"}:
                return np.array(nums[:9], dtype=np.float64).reshape(3, 3)

    for key in ["P2", "P0", "P1", "P3"]:
        if key in projections:
            return projections[key][:3, :3].copy()

    print("[WARN] Could not parse calibration. Using KITTI default K.")
    return K_default


def maybe_scale_intrinsics(
    K: np.ndarray,
    video_width: int,
    video_height: int,
    original_width: Optional[int],
    original_height: Optional[int],
) -> np.ndarray:
    K = K.copy().astype(np.float64)

    if original_width is None or original_height is None:
        return K

    sx = video_width / float(original_width)
    sy = video_height / float(original_height)

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    return K


def read_groundtruth(
    gt_file: Optional[Path],
    times: List[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Supported formats:
        KITTI: 12 numbers per line
        TUM: stamp tx ty tz qx qy qz qw
        Simple: tx ty tz or stamp tx ty tz
    """

    if gt_file is None or not gt_file.exists():
        return None, None

    lines = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            nums = read_numbers_from_line(line)
            if nums:
                lines.append(nums)

    if not lines:
        return None, None

    gt_times = []
    gt_xyz = []

    for i, nums in enumerate(lines):
        if len(nums) == 12:
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(nums, dtype=np.float64).reshape(3, 4)

            stamp = times[i] if i < len(times) else float(i)
            xyz = T[:3, 3]

        elif len(nums) >= 8:
            stamp = float(nums[0])
            xyz = np.array(nums[1:4], dtype=np.float64)

        elif len(nums) >= 4:
            stamp = float(nums[0])
            xyz = np.array(nums[1:4], dtype=np.float64)

        elif len(nums) >= 3:
            stamp = times[i] if i < len(times) else float(i)
            xyz = np.array(nums[:3], dtype=np.float64)

        else:
            continue

        gt_times.append(stamp)
        gt_xyz.append(xyz)

    if not gt_xyz:
        return None, None

    return np.array(gt_times), np.vstack(gt_xyz)


# ============================================================
# Rotation helpers
# ============================================================

def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)

    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ============================================================
# CNN Deep Motion Prior
# ============================================================

if TORCH_AVAILABLE:
    class SEBlock(nn.Module):
        """Lightweight squeeze-and-excitation attention block."""

        def __init__(self, channels: int, reduction: int = 8):
            super().__init__()
            hidden = max(channels // reduction, 8)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(self.pool(x))


    class ResidualConvBlock(nn.Module):
        """Residual CNN block with optional downsampling and channel attention."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.attn = SEBlock(out_ch)
            self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

            if stride != 1 or in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            identity = self.skip(x)
            out = F.silu(self.bn1(self.conv1(x)), inplace=True)
            out = self.drop(out)
            out = self.bn2(self.conv2(out))
            out = self.attn(out)
            return F.silu(out + identity, inplace=True)


    class CNNMotionPrior(nn.Module):
        """
        Residual attention CNN for online monocular scale regularization.

        Input:
            x: B x 3 x H x W
               channel 0 = normalized previous grayscale frame
               channel 1 = normalized current grayscale frame
               channel 2 = normalized signed frame difference

        Output:
            scale: positive metric frame-to-frame scale estimate
            log_var: predicted log-variance in log-scale space

        Why this is more accurate than the previous model:
            - residual blocks stabilize online training;
            - SE attention emphasizes motion-sensitive channels/regions;
            - the frame-difference channel makes the scale regression easier;
            - uncertainty lets the navigation filter reject weak prior estimates.
        """

        def __init__(self, min_scale: float = 1e-4, max_scale: float = 10.0):
            super().__init__()
            self.min_log_scale = float(math.log(max(min_scale, 1e-8)))
            self.max_log_scale = float(math.log(max(max_scale, min_scale + 1e-8)))

            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
            )

            self.features = nn.Sequential(
                ResidualConvBlock(32, 32, stride=1, dropout=0.02),
                ResidualConvBlock(32, 64, stride=2, dropout=0.03),
                ResidualConvBlock(64, 64, stride=1, dropout=0.03),
                ResidualConvBlock(64, 96, stride=2, dropout=0.04),
                ResidualConvBlock(96, 128, stride=2, dropout=0.05),
                ResidualConvBlock(128, 160, stride=2, dropout=0.05),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(160, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, 48),
                nn.SiLU(inplace=True),
                nn.Linear(48, 2),
            )

        def forward(self, x):
            z = self.stem(x)
            z = self.features(z)
            raw = self.head(z)

            log_scale = torch.clamp(
                raw[:, 0:1],
                min=self.min_log_scale,
                max=self.max_log_scale,
            )
            log_var = torch.clamp(raw[:, 1:2], min=-6.0, max=2.0)
            scale = torch.exp(log_scale)
            return scale, log_var


class DeepPriorWrapper:
    def __init__(
        self,
        enabled: bool,
        image_size: Tuple[int, int],
        lr: float,
        batch_size: int,
        buffer_size: int,
        min_train_samples: int,
        min_scale: float,
        max_scale: float,
        ckpt: Optional[Path],
        use_amp: bool = True,
        torch_compile_model: bool = False,
        ema_alpha: float = 0.30,
        aug_noise_std: float = 0.015,
    ):
        self.enabled = enabled and TORCH_AVAILABLE
        self.image_h, self.image_w = image_size
        self.batch_size = batch_size
        self.min_train_samples = min_train_samples
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 0.95))
        self.aug_noise_std = float(max(aug_noise_std, 0.0))

        self.buffer = deque(maxlen=buffer_size)
        self.train_steps = 0
        self.last_loss = float("nan")
        self.last_uncertainty = float("nan")
        self.last_pred_raw = float("nan")
        self.last_pred_ema = None

        if not self.enabled:
            self.model = None
            self.optimizer = None
            self.device = None
            self.use_amp = False
            self.scaler = None

            if enabled and not TORCH_AVAILABLE:
                print("[WARN] PyTorch is not available. CNN prior disabled.")

            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.model = CNNMotionPrior(min_scale=min_scale, max_scale=max_scale).to(self.device)

        if torch_compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("[INFO] torch.compile enabled for CNN prior.")
            except Exception as e:
                print(f"[WARN] torch.compile failed; continuing without it: {e}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=2e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if ckpt is not None and ckpt.exists():
            self._load_checkpoint_safely(ckpt)

    def _load_checkpoint_safely(self, ckpt: Path):
        """Load only compatible weights so older checkpoints do not crash the improved model."""
        try:
            state = torch.load(str(ckpt), map_location=self.device)
            model_state = state.get("model", state)
            current = self.model.state_dict()

            compatible = {}
            skipped = 0
            for key, value in model_state.items():
                clean_key = key.replace("_orig_mod.", "")
                if clean_key in current and current[clean_key].shape == value.shape:
                    compatible[clean_key] = value
                else:
                    skipped += 1

            if compatible:
                current.update(compatible)
                self.model.load_state_dict(current, strict=True)
                print(
                    f"[INFO] Loaded {len(compatible)} compatible CNN prior tensors from {ckpt}; "
                    f"skipped {skipped} tensors."
                )
            else:
                print(
                    f"[WARN] Existing prior checkpoint is incompatible with the improved model: {ckpt}. "
                    "Training a new prior from the online buffer."
                )

            if "optimizer" in state and skipped == 0:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                except Exception:
                    print("[WARN] Optimizer state could not be loaded; using a fresh optimizer.")

            self.train_steps = int(state.get("train_steps", 0))
            if "last_pred_ema" in state:
                self.last_pred_ema = float(state["last_pred_ema"])

        except Exception as e:
            print(f"[WARN] Could not load CNN prior checkpoint {ckpt}: {e}")

    def frame_pair_to_tensor(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        prev = cv2.resize(prev_gray, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)
        curr = cv2.resize(curr_gray, (self.image_w, self.image_h), interpolation=cv2.INTER_AREA)

        prev_raw = prev.astype(np.float32) / 255.0
        curr_raw = curr.astype(np.float32) / 255.0
        diff_raw = curr_raw - prev_raw

        def standardize(img: np.ndarray) -> np.ndarray:
            return (img - float(img.mean())) / (float(img.std()) + 1e-6)

        prev_n = standardize(prev_raw)
        curr_n = standardize(curr_raw)
        diff_n = standardize(diff_raw)

        x = np.stack([prev_n, curr_n, diff_n], axis=0)
        tensor = torch.from_numpy(x).float()

        if self.enabled and self.device is not None and self.device.type == "cuda":
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass

        return tensor

    def add_sample(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        target_scale: Optional[float],
    ):
        if not self.enabled:
            return

        if target_scale is None:
            return

        if not np.isfinite(target_scale):
            return

        if not (self.min_scale <= target_scale <= self.max_scale):
            return

        with torch.no_grad():
            x = self.frame_pair_to_tensor(prev_gray, curr_gray)

        y = float(np.clip(target_scale, self.min_scale, self.max_scale))
        log_y = float(np.log(y + 1e-8))
        self.buffer.append((x, y, log_y))

    def _augment_batch(self, xs):
        """Small online augmentations that preserve frame-to-frame metric scale."""
        if xs.ndim != 4:
            return xs

        # Horizontal image flip preserves the scalar displacement magnitude.
        if random.random() < 0.50:
            xs = torch.flip(xs, dims=[-1])

        # Mild noise prevents overfitting to a few early frames.
        if self.aug_noise_std > 0 and random.random() < 0.75:
            xs = xs + self.aug_noise_std * torch.randn_like(xs)

        return xs

    def train_once(self, steps: int = 1):
        if not self.enabled:
            return

        if len(self.buffer) < self.min_train_samples:
            return

        self.model.train()

        for _ in range(steps):
            batch_size = min(self.batch_size, len(self.buffer))
            batch = random.sample(list(self.buffer), batch_size)

            xs = torch.stack([b[0] for b in batch], dim=0).to(
                self.device,
                non_blocking=True,
            )
            ys = torch.tensor([b[1] for b in batch], dtype=torch.float32).view(-1, 1).to(
                self.device,
                non_blocking=True,
            )
            log_ys = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1).to(
                self.device,
                non_blocking=True,
            )

            xs = self._augment_batch(xs)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred, log_var = self.model(xs)
                pred_log = torch.log(pred + 1e-8)

                # Robust scale regression in log space.
                robust_loss = F.smooth_l1_loss(pred_log, log_ys, beta=0.20)

                # Heteroscedastic uncertainty term: the model learns to reduce
                # confidence on ambiguous frames instead of producing overconfident jumps.
                sq_error = (pred_log - log_ys) ** 2
                uncertainty_loss = torch.mean(torch.exp(-log_var) * sq_error + 0.02 * log_var)

                # A tiny linear-domain term improves convergence when scale is close to zero.
                linear_loss = F.smooth_l1_loss(pred, ys, beta=0.05)

                loss = robust_loss + 0.15 * uncertainty_loss + 0.05 * linear_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.last_loss = float(loss.detach().cpu())
            self.train_steps += 1

    def predict_scale(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Tuple[Optional[float], float]:
        if not self.enabled:
            return None, 0.0

        if len(self.buffer) < self.min_train_samples and self.train_steps < self.min_train_samples:
            return None, 0.0

        self.model.eval()

        with torch.no_grad():
            x = self.frame_pair_to_tensor(prev_gray, curr_gray)
            x = x.unsqueeze(0).to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred, log_var = self.model(x)
                pred_value = float(pred.item())
                log_var_value = float(log_var.item())

        pred_value = float(np.clip(pred_value, self.min_scale, self.max_scale))
        self.last_pred_raw = pred_value

        if self.last_pred_ema is None or not np.isfinite(self.last_pred_ema):
            self.last_pred_ema = pred_value
        else:
            # Do not let a single abnormal prediction corrupt the EMA.
            ratio = pred_value / max(self.last_pred_ema, 1e-8)
            if 0.20 <= ratio <= 5.0:
                self.last_pred_ema = (
                    (1.0 - self.ema_alpha) * self.last_pred_ema
                    + self.ema_alpha * pred_value
                )
            else:
                self.last_pred_ema = 0.98 * self.last_pred_ema + 0.02 * pred_value

        pred_smooth = float(np.clip(self.last_pred_ema, self.min_scale, self.max_scale))

        # Convert predicted log variance to a practical confidence factor.
        sigma_log = float(np.sqrt(max(np.exp(log_var_value), 1e-8)))
        uncertainty_conf = float(np.clip(np.exp(-1.75 * sigma_log), 0.05, 1.0))
        maturity_conf = min(1.0, self.train_steps / float(max(self.min_train_samples * 3, 1)))
        buffer_conf = min(1.0, len(self.buffer) / float(max(self.min_train_samples * 2, 1)))
        confidence = float(np.clip(maturity_conf * buffer_conf * uncertainty_conf, 0.0, 1.0))

        self.last_uncertainty = sigma_log
        return pred_smooth, confidence

    def save(self, ckpt: Path):
        if not self.enabled:
            return

        ckpt.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "last_pred_ema": self.last_pred_ema,
                "model_type": "ResidualAttentionCNNMotionPriorV2",
            },
            str(ckpt),
        )


# ============================================================
# Classical VO
# ============================================================

def preprocess_gray(
    frame: np.ndarray,
    use_clahe: bool = True,
    use_cuda: bool = False,
) -> np.ndarray:
    """
    CPU/GPU grayscale preprocessing.

    GPU path requires OpenCV built with CUDA support. If CUDA preprocessing
    fails, the function falls back to CPU automatically.
    """

    if use_cuda and CUDA_CV_AVAILABLE:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            if frame.ndim == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            else:
                gpu_gray = gpu_frame

            if use_clahe:
                clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gpu_gray = clahe.apply(gpu_gray)

            return gpu_gray.download()

        except Exception:
            # Safe fallback for OpenCV builds without full CUDA image ops.
            pass

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray


def estimate_relative_pose_orb(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    K: np.ndarray,
    orb_cpu,
    matcher: cv2.BFMatcher,
    ransac_thresh: float,
    ratio: float,
    min_matches: int,
    use_cuda_orb: bool = False,
    orb_cuda=None,
):
    """
    Estimate relative pose using ORB + Essential matrix.

    If use_cuda_orb is True and OpenCV CUDA ORB is available, ORB detection
    and descriptor extraction are attempted on GPU. Descriptor matching and
    Essential-matrix estimation remain CPU-side because they are more portable
    across OpenCV builds. Automatic fallback to CPU ORB is included.
    """

    info = {
        "kp1": 0,
        "kp2": 0,
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "pts2": None,
        "gpu_orb": 0,
    }

    kp1 = kp2 = None
    des1 = des2 = None

    if use_cuda_orb and orb_cuda is not None and CUDA_ORB_AVAILABLE:
        try:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            gpu_prev.upload(prev_gray)
            gpu_curr.upload(curr_gray)

            kp1_gpu, des1_gpu = orb_cuda.detectAndComputeAsync(gpu_prev, None)
            kp2_gpu, des2_gpu = orb_cuda.detectAndComputeAsync(gpu_curr, None)

            kp1 = orb_cuda.convert(kp1_gpu)
            kp2 = orb_cuda.convert(kp2_gpu)

            if des1_gpu is not None and des2_gpu is not None:
                des1 = des1_gpu.download()
                des2 = des2_gpu.download()
                info["gpu_orb"] = 1

        except Exception:
            kp1 = kp2 = None
            des1 = des2 = None
            info["gpu_orb"] = 0

    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        kp1, des1 = orb_cpu.detectAndCompute(prev_gray, None)
        kp2, des2 = orb_cpu.detectAndCompute(curr_gray, None)
        info["gpu_orb"] = 0

    info["kp1"] = len(kp1) if kp1 is not None else 0
    info["kp2"] = len(kp2) if kp2 is not None else 0

    if des1 is None or des2 is None:
        return None, None, info

    if len(kp1) < min_matches or len(kp2) < min_matches:
        return None, None, info

    raw = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)
    info["matches"] = len(good)

    if len(good) < min_matches:
        return None, None, info

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh,
    )

    if E is None or mask is None:
        return None, None, info

    if E.shape[0] > 3:
        E = E[:3, :3]

    inlier_mask = mask.ravel().astype(bool)

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    if len(pts1_in) < min_matches:
        return None, None, info

    inliers, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)

    if inliers < min_matches:
        return None, None, info

    pose_mask = pose_mask.ravel() > 0

    info["inliers"] = int(inliers)
    info["inlier_ratio"] = float(inliers) / max(float(len(good)), 1.0)
    info["pts2"] = pts2_in[pose_mask]

    return R.astype(np.float64), t.reshape(3).astype(np.float64), info


def gt_frame_scale(
    frame_idx: int,
    gt_xyz: Optional[np.ndarray],
    min_scale: float,
    max_scale: float,
) -> Optional[float]:
    if gt_xyz is None:
        return None

    if not (0 < frame_idx < len(gt_xyz)):
        return None

    scale = float(np.linalg.norm(gt_xyz[frame_idx] - gt_xyz[frame_idx - 1]))

    if min_scale <= scale <= max_scale:
        return scale

    return None


def get_base_scale(
    frame_idx: int,
    gt_xyz: Optional[np.ndarray],
    scale_mode: str,
    default_scale: float,
    min_scale: float,
    max_scale: float,
) -> float:
    use_gt = scale_mode == "gt" or (scale_mode == "auto" and gt_xyz is not None)

    if use_gt:
        s = gt_frame_scale(frame_idx, gt_xyz, min_scale, max_scale)

        if s is not None:
            return s

    return default_scale


def blend_scale_with_cnn(
    base_scale: float,
    cnn_scale: Optional[float],
    cnn_conf: float,
    prior_blend: float,
    min_scale: float,
    max_scale: float,
) -> Tuple[float, float]:
    if cnn_scale is None or cnn_conf <= 0:
        return base_scale, 0.0

    ratio = cnn_scale / max(base_scale, 1e-6)

    if ratio < 0.30 or ratio > 3.00:
        alpha = 0.0

    elif ratio < 0.50 or ratio > 2.00:
        alpha = 0.25 * prior_blend * cnn_conf

    else:
        alpha = prior_blend * cnn_conf

    scale = (1.0 - alpha) * base_scale + alpha * cnn_scale
    scale = float(np.clip(scale, min_scale, max_scale))

    return scale, alpha


class MotionSmoother:
    """
    Constant-velocity motion prior for reducing jitter and bad jumps.
    """

    def __init__(self, alpha: float):
        self.alpha = float(np.clip(alpha, 0.0, 0.95))
        self.prev_step = None

    def filter(self, measured_step: np.ndarray, inlier_ratio: float) -> np.ndarray:
        measured_step = np.asarray(measured_step, dtype=np.float64)

        if self.prev_step is None:
            self.prev_step = measured_step.copy()
            return measured_step

        confidence = float(np.clip(inlier_ratio, 0.0, 1.0))
        adaptive_alpha = self.alpha * (1.0 - 0.5 * confidence)

        predicted_step = self.prev_step
        filtered = (1.0 - adaptive_alpha) * measured_step + adaptive_alpha * predicted_step

        self.prev_step = filtered.copy()

        return filtered


class ParticleFilter3D:
    """
    Bootstrap particle filter for 3D VO position smoothing.

    State per particle:
        [x, y, z, vx, vy, vz]

    The filter uses:
        - control input: VO/CNN-prior step vector
        - measurement: accumulated raw VO position
        - quality: inlier-ratio/CNN-confidence-based reliability

    It does not use ground truth for filtering. GT is used only when
    --scale_mode auto/gt is selected for monocular scale recovery, as in
    the original script.
    """

    def __init__(
        self,
        n_particles: int,
        init_pos: np.ndarray,
        process_noise: float,
        measurement_noise: float,
        velocity_noise: float,
        resample_threshold: float,
        seed: int,
    ):
        self.n_particles = int(max(n_particles, 100))
        self.process_noise = float(max(process_noise, 1e-6))
        self.measurement_noise_base = float(max(measurement_noise, 1e-6))
        self.velocity_noise = float(max(velocity_noise, 1e-6))
        self.resample_threshold = float(np.clip(resample_threshold, 0.05, 0.95))
        self.rng = np.random.default_rng(seed)

        init_pos = np.asarray(init_pos, dtype=np.float64).reshape(3)

        self.particles = np.zeros((self.n_particles, 6), dtype=np.float64)
        self.particles[:, 0:3] = init_pos[None, :] + self.rng.normal(
            0.0,
            self.measurement_noise_base * 0.20,
            size=(self.n_particles, 3),
        )
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)

        self.last_neff = float(self.n_particles)
        self.last_quality = 0.0
        self.last_measurement_noise = self.measurement_noise_base
        self.last_resampled = 0

    def effective_sample_size(self) -> float:
        return float(1.0 / max(np.sum(self.weights ** 2), 1e-300))

    def estimate(self) -> np.ndarray:
        return np.average(self.particles[:, 0:3], axis=0, weights=self.weights)

    def predict(self, control_step: np.ndarray, dt: float, quality: float):
        control_step = np.asarray(control_step, dtype=np.float64).reshape(3)

        dt = float(max(dt, 1e-3))
        quality = float(np.clip(quality, 0.0, 1.0))
        step_norm = float(np.linalg.norm(control_step))

        # When quality is low, allow more spread. When quality is high,
        # keep particles concentrated around the VO/CNN motion step.
        q_pos = self.process_noise * (1.35 - 0.65 * quality) * max(1.0, step_norm)
        q_vel = self.velocity_noise * (1.40 - 0.60 * quality)

        measured_velocity = control_step / dt

        # Predict position using the visual motion increment plus noise.
        self.particles[:, 0:3] += control_step[None, :]
        self.particles[:, 0:3] += self.rng.normal(0.0, q_pos, size=(self.n_particles, 3))

        # Blend particle velocities toward the current VO velocity.
        beta = 0.25 + 0.45 * quality
        self.particles[:, 3:6] = (
            (1.0 - beta) * self.particles[:, 3:6]
            + beta * measured_velocity[None, :]
            + self.rng.normal(0.0, q_vel, size=(self.n_particles, 3))
        )

    def update(self, measurement_pos: Optional[np.ndarray], quality: float):
        self.last_quality = float(np.clip(quality, 0.0, 1.0))
        self.last_resampled = 0

        if measurement_pos is None:
            self.last_neff = self.effective_sample_size()
            return

        measurement_pos = np.asarray(measurement_pos, dtype=np.float64).reshape(3)

        # Adaptive measurement noise:
        # high feature quality -> trust raw VO position more,
        # low feature quality -> trust particle prediction more.
        r = self.measurement_noise_base / (0.25 + 0.75 * self.last_quality)
        r = float(max(r, 1e-6))
        self.last_measurement_noise = r

        diff = self.particles[:, 0:3] - measurement_pos[None, :]
        d2 = np.sum(diff * diff, axis=1)

        likelihood = np.exp(-0.5 * d2 / (r * r)) + 1e-300
        self.weights *= likelihood

        weight_sum = float(np.sum(self.weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= weight_sum

        self.last_neff = self.effective_sample_size()

        if self.last_neff < self.resample_threshold * self.n_particles:
            self.resample_systematic()
            self.last_resampled = 1

    def resample_systematic(self):
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0

        indexes = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.n_particles)

        # Small roughening after resampling prevents sample impoverishment.
        rough_pos = self.process_noise * 0.10
        rough_vel = self.velocity_noise * 0.10
        self.particles[:, 0:3] += self.rng.normal(0.0, rough_pos, size=(self.n_particles, 3))
        self.particles[:, 3:6] += self.rng.normal(0.0, rough_vel, size=(self.n_particles, 3))

        self.last_neff = float(self.n_particles)

    def step(
        self,
        measurement_pos: Optional[np.ndarray],
        control_step: np.ndarray,
        dt: float,
        quality: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        self.predict(control_step=control_step, dt=dt, quality=quality)
        self.update(measurement_pos=measurement_pos, quality=quality)

        return self.estimate(), self.info()

    def info(self) -> Dict[str, float]:
        return {
            "particle_neff": float(self.last_neff),
            "particle_quality": float(self.last_quality),
            "particle_measurement_noise": float(self.last_measurement_noise),
            "particle_resampled": int(self.last_resampled),
            "particle_count": int(self.n_particles),
        }


# ============================================================
# Evaluation and plotting
# ============================================================

def umeyama_align(src: np.ndarray, dst: np.ndarray):
    """
    Similarity alignment:
        dst ~= s * R * src + t
    """

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    n = min(len(src), len(dst))
    src = src[:n]
    dst = dst[:n]

    if n < 3:
        return src.copy(), 1.0

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    var_src = np.mean(np.sum(X * X, axis=1))

    if var_src < 1e-12:
        return src.copy(), 1.0

    cov = (Y.T @ X) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - scale * (R @ mu_src)

    aligned = (scale * (R @ src.T)).T + t

    return aligned, scale


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))

    if n == 0:
        return float("nan")

    e = a[:n] - b[:n]
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def compute_trajectory_error_report(
    vo_xyz: np.ndarray,
    gt_xyz: Optional[np.ndarray],
    times: List[float],
) -> Dict[str, object]:
    """
    Computes trajectory errors after similarity alignment.

    Returns:
        dict with aligned trajectory, per-frame error table, and summary metrics.

    Metrics:
        - ATE RMSE: sqrt(mean(||p_est - p_gt||^2))
        - ATE MAE: mean(||p_est - p_gt||)
        - ATE median/max/final
        - per-axis RMSE
        - x-z plane RMSE
    """

    report = {
        "available": False,
        "n": 0,
        "aligned": None,
        "sim_scale": float("nan"),
        "ate_rmse": float("nan"),
        "ate_mae": float("nan"),
        "ate_median": float("nan"),
        "ate_max": float("nan"),
        "ate_final": float("nan"),
        "rmse_x": float("nan"),
        "rmse_y": float("nan"),
        "rmse_z": float("nan"),
        "rmse_xz": float("nan"),
        "error_rows": [],
    }

    if gt_xyz is None or len(gt_xyz) < 3 or len(vo_xyz) < 3:
        return report

    n = min(len(vo_xyz), len(gt_xyz), len(times))
    if n < 3:
        return report

    aligned, sim_scale = umeyama_align(vo_xyz[:n], gt_xyz[:n])

    err = aligned - gt_xyz[:n]
    abs_err = np.linalg.norm(err, axis=1)
    xz_err = np.linalg.norm(err[:, [0, 2]], axis=1)

    report["available"] = True
    report["n"] = int(n)
    report["aligned"] = aligned
    report["sim_scale"] = float(sim_scale)
    report["ate_rmse"] = float(np.sqrt(np.mean(abs_err ** 2)))
    report["ate_mae"] = float(np.mean(abs_err))
    report["ate_median"] = float(np.median(abs_err))
    report["ate_max"] = float(np.max(abs_err))
    report["ate_final"] = float(abs_err[-1])
    report["rmse_x"] = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    report["rmse_y"] = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    report["rmse_z"] = float(np.sqrt(np.mean(err[:, 2] ** 2)))
    report["rmse_xz"] = float(np.sqrt(np.mean(xz_err ** 2)))

    rows = []
    for i in range(n):
        rows.append(
            {
                "frame": int(i),
                "time": float(times[i]),
                "est_x": float(vo_xyz[i, 0]),
                "est_y": float(vo_xyz[i, 1]),
                "est_z": float(vo_xyz[i, 2]),
                "aligned_x": float(aligned[i, 0]),
                "aligned_y": float(aligned[i, 1]),
                "aligned_z": float(aligned[i, 2]),
                "gt_x": float(gt_xyz[i, 0]),
                "gt_y": float(gt_xyz[i, 1]),
                "gt_z": float(gt_xyz[i, 2]),
                "error_x": float(err[i, 0]),
                "error_y": float(err[i, 1]),
                "error_z": float(err[i, 2]),
                "abs_error": float(abs_err[i]),
                "xz_error": float(xz_err[i]),
            }
        )

    report["error_rows"] = rows

    return report


def compute_online_error_stats(
    vo_xyz_list: List[np.ndarray],
    gt_xyz: Optional[np.ndarray],
    times: List[float],
    frame_idx: int,
) -> Dict[str, object]:
    """
    Computes online/display-only error metrics up to the current frame.

    Important:
        This is used only for visualization in the video/map.
        It does not modify the VO trajectory, particle filter, or CNN prior.
    """

    stats = {
        "available": False,
        "n": 0,
        "current_error": float("nan"),
        "current_xz_error": float("nan"),
        "running_rmse": float("nan"),
        "running_rmse_xz": float("nan"),
        "running_mae": float("nan"),
        "max_error": float("nan"),
        "sim_scale": float("nan"),
        "history_abs_error": [],
        "history_xz_error": [],
        "history_time": [],
    }

    if gt_xyz is None:
        return stats

    if vo_xyz_list is None or len(vo_xyz_list) < 3 or len(gt_xyz) < 3:
        return stats

    vo_np = np.asarray(vo_xyz_list, dtype=np.float64)

    n = min(len(vo_np), len(gt_xyz), frame_idx + 1, len(times))

    if n < 3:
        return stats

    aligned, sim_scale = umeyama_align(vo_np[:n], gt_xyz[:n])

    err = aligned - gt_xyz[:n]
    abs_err = np.linalg.norm(err, axis=1)
    xz_err = np.linalg.norm(err[:, [0, 2]], axis=1)

    stats["available"] = True
    stats["n"] = int(n)
    stats["current_error"] = float(abs_err[-1])
    stats["current_xz_error"] = float(xz_err[-1])
    stats["running_rmse"] = float(np.sqrt(np.mean(abs_err ** 2)))
    stats["running_rmse_xz"] = float(np.sqrt(np.mean(xz_err ** 2)))
    stats["running_mae"] = float(np.mean(abs_err))
    stats["max_error"] = float(np.max(abs_err))
    stats["sim_scale"] = float(sim_scale)
    stats["history_abs_error"] = abs_err.tolist()
    stats["history_xz_error"] = xz_err.tolist()
    stats["history_time"] = list(times[:n])

    return stats


def save_error_csv(out_file: Path, error_rows: List[Dict[str, float]]):
    if not error_rows:
        return

    fieldnames = [
        "frame",
        "time",
        "est_x",
        "est_y",
        "est_z",
        "aligned_x",
        "aligned_y",
        "aligned_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "error_x",
        "error_y",
        "error_z",
        "abs_error",
        "xz_error",
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)


def save_error_plot(out_file: Path, error_rows: List[Dict[str, float]]):
    if not error_rows:
        return

    t = np.array([r["time"] for r in error_rows], dtype=np.float64)
    abs_error = np.array([r["abs_error"] for r in error_rows], dtype=np.float64)
    xz_error = np.array([r["xz_error"] for r in error_rows], dtype=np.float64)

    plt.figure(figsize=(9, 5))
    plt.plot(t, abs_error, label="3D position error", linewidth=2)
    plt.plot(t, xz_error, label="x-z error", linewidth=2)
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("Trajectory error over time")
    plt.grid(False)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def fill_record_errors(
    records: List[Dict[str, float]],
    error_rows: List[Dict[str, float]],
):
    """
    Adds aligned position, GT position, and error values to trajectory.csv rows.
    Rows without GT receive NaN values.
    """

    default_error_values = {
        "gt_x": float("nan"),
        "gt_y": float("nan"),
        "gt_z": float("nan"),
        "aligned_x": float("nan"),
        "aligned_y": float("nan"),
        "aligned_z": float("nan"),
        "error_x": float("nan"),
        "error_y": float("nan"),
        "error_z": float("nan"),
        "abs_error": float("nan"),
        "xz_error": float("nan"),
    }

    for row in records:
        row.update(default_error_values)

    for row in error_rows:
        idx = int(row["frame"])
        if 0 <= idx < len(records):
            records[idx].update(
                {
                    "gt_x": row["gt_x"],
                    "gt_y": row["gt_y"],
                    "gt_z": row["gt_z"],
                    "aligned_x": row["aligned_x"],
                    "aligned_y": row["aligned_y"],
                    "aligned_z": row["aligned_z"],
                    "error_x": row["error_x"],
                    "error_y": row["error_y"],
                    "error_z": row["error_z"],
                    "abs_error": row["abs_error"],
                    "xz_error": row["xz_error"],
                }
            )


def save_trajectory_plot(
    out_file: Path,
    vo_xyz: np.ndarray,
    gt_xyz: Optional[np.ndarray] = None,
    title: str = "Monocular VO trajectory",
):
    plt.figure(figsize=(8, 6))

    plt.plot(
        vo_xyz[:, 0],
        vo_xyz[:, 2],
        label="VO",
        linewidth=2,
    )

    if gt_xyz is not None:
        n = min(len(vo_xyz), len(gt_xyz))
        plt.plot(
            gt_xyz[:n, 0],
            gt_xyz[:n, 2],
            label="GT",
            linewidth=2,
        )

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(title)

    # Updated: no grid in saved graphs.
    plt.grid(False)

    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


# ============================================================
# Dashboard drawing
# ============================================================

def draw_feature_overlay(frame_bgr: np.ndarray, pts: Optional[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()

    if pts is None:
        return out

    for p in pts.astype(int):
        cv2.circle(
            out,
            (int(p[0]), int(p[1])),
            2,
            (0, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

    return out


def draw_error_panel_on_map(
    canvas: np.ndarray,
    error_stats: Optional[Dict[str, object]],
    size: int,
) -> np.ndarray:
    """
    Draws online RMSE/error values and a small error curve on the trajectory map.
    """

    if error_stats is None or not error_stats.get("available", False):
        cv2.putText(
            canvas,
            "RMSE: waiting for GT alignment",
            (15, size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (60, 60, 60),
            1,
            lineType=cv2.LINE_AA,
        )
        return canvas

    panel_x1 = 12
    panel_y1 = size - 125
    panel_x2 = size - 12
    panel_y2 = size - 12

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (255, 255, 255),
        -1,
    )
    canvas[:] = cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0)

    cv2.rectangle(
        canvas,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (180, 180, 180),
        1,
        lineType=cv2.LINE_AA,
    )

    running_rmse = float(error_stats.get("running_rmse", float("nan")))
    running_rmse_xz = float(error_stats.get("running_rmse_xz", float("nan")))
    current_error = float(error_stats.get("current_error", float("nan")))
    current_xz_error = float(error_stats.get("current_xz_error", float("nan")))
    running_mae = float(error_stats.get("running_mae", float("nan")))
    max_error = float(error_stats.get("max_error", float("nan")))

    cv2.putText(
        canvas,
        f"ATE RMSE: {running_rmse:.3f} m | x-z RMSE: {running_rmse_xz:.3f} m",
        (panel_x1 + 10, panel_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (25, 25, 25),
        1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Current: {current_error:.3f} m | x-z: {current_xz_error:.3f} m | MAE: {running_mae:.3f} m | Max: {max_error:.3f} m",
        (panel_x1 + 10, panel_y1 + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (25, 25, 25),
        1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Mini error graph
    # --------------------------------------------------------
    history_abs = np.asarray(
        error_stats.get("history_abs_error", []),
        dtype=np.float64,
    ).reshape(-1)

    history_xz = np.asarray(
        error_stats.get("history_xz_error", []),
        dtype=np.float64,
    ).reshape(-1)

    if len(history_abs) >= 2:
        gx1 = panel_x1 + 12
        gy1 = panel_y1 + 63
        gx2 = panel_x2 - 12
        gy2 = panel_y2 - 10

        cv2.rectangle(
            canvas,
            (gx1, gy1),
            (gx2, gy2),
            (210, 210, 210),
            1,
            lineType=cv2.LINE_AA,
        )

        max_e = float(max(np.max(history_abs), np.max(history_xz), 1e-6))

        def make_curve(values: np.ndarray):
            pts = []

            for i, value in enumerate(values):
                x = gx1 + int(round(i * (gx2 - gx1) / max(len(values) - 1, 1)))
                y = gy2 - int(round((float(value) / max_e) * (gy2 - gy1)))
                pts.append((x, y))

            return pts

        abs_pts = make_curve(history_abs)
        xz_pts = make_curve(history_xz)

        for a, b in zip(abs_pts[:-1], abs_pts[1:]):
            cv2.line(canvas, a, b, (0, 120, 255), 2, lineType=cv2.LINE_AA)

        for a, b in zip(xz_pts[:-1], xz_pts[1:]):
            cv2.line(canvas, a, b, (80, 180, 80), 1, lineType=cv2.LINE_AA)

        cv2.putText(
            canvas,
            "orange: 3D error | green: x-z error",
            (gx1 + 4, gy1 + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (50, 50, 50),
            1,
            lineType=cv2.LINE_AA,
        )

    return canvas


def draw_trajectory_map(
    vo_xyz: List[np.ndarray],
    gt_xyz: Optional[np.ndarray],
    frame_idx: int,
    size: int,
    meters_to_pixels: float,
    error_stats: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """
    Draws the x-z trajectory map for the output video.

    Updates:
        1. Map content is centered horizontally and vertically.
        2. Grid removed.
        3. If GT exists, VO is aligned to GT for display using Umeyama.
        4. Raw VO trajectory is not modified.
        5. Legend is placed in the upper-right corner and clearly defines VO and GT.
        6. Online RMSE/error values are drawn in the bottom map panel.
    """

    _ = meters_to_pixels  # Kept for compatibility with earlier calls.

    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    margin = 40

    vo_np = np.asarray(vo_xyz, dtype=np.float64)
    if len(vo_np) == 0:
        return canvas

    # --------------------------------------------------------
    # Display-only VO alignment to GT
    # --------------------------------------------------------
    display_vo = vo_np.copy()
    display_gt = None

    if gt_xyz is not None and len(gt_xyz) > 1:
        end_gt = min(frame_idx + 1, len(gt_xyz))
        display_gt = gt_xyz[:end_gt].copy()

        n_align = min(len(display_vo), len(display_gt))
        if n_align >= 3:
            display_vo, _ = umeyama_align(display_vo[:n_align], display_gt[:n_align])
        else:
            display_vo = display_vo[:n_align]

    # --------------------------------------------------------
    # Stable bounds for the whole map
    # --------------------------------------------------------
    if gt_xyz is not None and len(gt_xyz) > 1:
        gt_bounds = gt_xyz[:, [0, 2]]
        vo_bounds = display_vo[:, [0, 2]] if len(display_vo) > 0 else gt_bounds
        bounds_pts = np.vstack([gt_bounds, vo_bounds])
    else:
        bounds_pts = display_vo[:, [0, 2]]

    min_x = float(np.min(bounds_pts[:, 0]))
    max_x = float(np.max(bounds_pts[:, 0]))
    min_z = float(np.min(bounds_pts[:, 1]))
    max_z = float(np.max(bounds_pts[:, 1]))

    range_x = max(max_x - min_x, 1e-6)
    range_z = max(max_z - min_z, 1e-6)

    pad_x = 0.10 * range_x
    pad_z = 0.10 * range_z

    min_x -= pad_x
    max_x += pad_x
    min_z -= pad_z
    max_z += pad_z

    range_x = max(max_x - min_x, 1e-6)
    range_z = max(max_z - min_z, 1e-6)

    drawable_w = size - 2 * margin
    drawable_h = size - 2 * margin

    scale_x = drawable_w / range_x
    scale_z = drawable_h / range_z
    scale = min(scale_x, scale_z)

    plot_w = range_x * scale
    plot_h = range_z * scale

    # Center the plotted content inside the drawable area.
    offset_x = margin + (drawable_w - plot_w) / 2.0
    offset_y = margin + (drawable_h - plot_h) / 2.0

    def project_xz(p: np.ndarray) -> Tuple[int, int]:
        x = int(round(offset_x + (p[0] - min_x) * scale))
        y = int(round(offset_y + (max_z - p[2]) * scale))
        return x, y

    # --------------------------------------------------------
    # Clean border
    # --------------------------------------------------------
    cv2.rectangle(
        canvas,
        (margin, margin),
        (size - margin, size - margin),
        (210, 210, 210),
        1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Draw GT trajectory
    # --------------------------------------------------------
    if display_gt is not None and len(display_gt) > 1:
        gt_pts = [project_xz(p) for p in display_gt]

        for a, b in zip(gt_pts[:-1], gt_pts[1:]):
            cv2.line(canvas, a, b, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        cv2.circle(canvas, gt_pts[-1], 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # --------------------------------------------------------
    # Draw aligned VO trajectory
    # --------------------------------------------------------
    if len(display_vo) > 1:
        vo_pts = [project_xz(p) for p in display_vo]

        for a, b in zip(vo_pts[:-1], vo_pts[1:]):
            cv2.line(canvas, a, b, (255, 0, 0), 2, lineType=cv2.LINE_AA)

        cv2.circle(canvas, vo_pts[-1], 5, (0, 180, 255), -1, lineType=cv2.LINE_AA)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    cv2.putText(
        canvas,
        "Trajectory map: x-z",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )

    if gt_xyz is not None:
        cv2.putText(
            canvas,
            "VO aligned to GT",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Legend: upper-right corner
    # --------------------------------------------------------
    legend_w = 120
    legend_h = 58
    legend_x1 = size - margin - legend_w - 8
    legend_y1 = margin + 8
    legend_x2 = legend_x1 + legend_w
    legend_y2 = legend_y1 + legend_h

    cv2.rectangle(canvas, (legend_x1, legend_y1), (legend_x2, legend_y2), (255, 255, 255), -1)
    cv2.rectangle(canvas, (legend_x1, legend_y1), (legend_x2, legend_y2), (190, 190, 190), 1)

    # VO legend
    cv2.line(
        canvas,
        (legend_x1 + 10, legend_y1 + 18),
        (legend_x1 + 40, legend_y1 + 18),
        (255, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "VO",
        (legend_x1 + 50, legend_y1 + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (30, 30, 30),
        1,
        lineType=cv2.LINE_AA,
    )

    # GT legend
    cv2.line(
        canvas,
        (legend_x1 + 10, legend_y1 + 42),
        (legend_x1 + 40, legend_y1 + 42),
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GT",
        (legend_x1 + 50, legend_y1 + 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (30, 30, 30),
        1,
        lineType=cv2.LINE_AA,
    )

    canvas = draw_error_panel_on_map(
        canvas=canvas,
        error_stats=error_stats,
        size=size,
    )

    return canvas


# ============================================================
# Output writers
# ============================================================

def write_outputs(
    outdir: Path,
    records: List[Dict[str, float]],
    poses: List[np.ndarray],
):
    csv_file = outdir / "trajectory.csv"
    tum_file = outdir / "trajectory_tum.txt"
    kitti_file = outdir / "trajectory_kitti.txt"

    fieldnames = [
        "frame",
        "time",
        "x",
        "y",
        "z",
        "gt_x",
        "gt_y",
        "gt_z",
        "aligned_x",
        "aligned_y",
        "aligned_z",
        "error_x",
        "error_y",
        "error_z",
        "abs_error",
        "xz_error",
        "raw_x",
        "raw_y",
        "raw_z",
        "particle_enabled",
        "particle_count",
        "particle_neff",
        "particle_quality",
        "particle_measurement_noise",
        "particle_resampled",
        "qx",
        "qy",
        "qz",
        "qw",
        "base_scale",
        "cnn_scale",
        "cnn_conf",
        "cnn_uncertainty",
        "cnn_pred_raw",
        "final_scale",
        "cnn_alpha",
        "kp_prev",
        "kp_curr",
        "matches",
        "inliers",
        "inlier_ratio",
        "gpu_orb",
        "prior_loss",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(tum_file, "w", encoding="utf-8") as f:
        for row in records:
            f.write(
                f"{row['time']:.9f} "
                f"{row['x']:.9f} {row['y']:.9f} {row['z']:.9f} "
                f"{row['qx']:.9f} {row['qy']:.9f} "
                f"{row['qz']:.9f} {row['qw']:.9f}\n"
            )

    with open(kitti_file, "w", encoding="utf-8") as f:
        for T in poses:
            vals = T[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.9f}" for v in vals) + "\n")


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="kitti06")
    parser.add_argument("--outdir", type=str, default="vo_output_cnn_prior_particle_gpu")

    parser.add_argument(
        "--scale_mode",
        type=str,
        default="auto",
        choices=["auto", "gt", "unit"],
    )

    parser.add_argument("--default_scale", type=float, default=1.0)
    parser.add_argument("--min_scale", type=float, default=1e-4)
    parser.add_argument("--max_scale", type=float, default=10.0)

    parser.add_argument("--nfeatures", type=int, default=7000)
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--ransac_thresh", type=float, default=0.75)
    parser.add_argument("--min_matches", type=int, default=50)

    parser.add_argument("--map_size", type=int, default=520)
    parser.add_argument(
        "--map_scale",
        type=float,
        default=15.0,
        help="Kept for compatibility. The updated video map uses auto-fit bounds.",
    )

    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--no_clahe", action="store_true")

    parser.add_argument("--invert_motion", action="store_true")

    parser.add_argument("--orig_width", type=int, default=None)
    parser.add_argument("--orig_height", type=int, default=None)

    parser.add_argument("--use_cnn_prior", action="store_true")
    parser.add_argument("--prior_blend", type=float, default=0.35)
    parser.add_argument("--prior_lr", type=float, default=1e-4)
    parser.add_argument("--prior_batch", type=int, default=8)
    parser.add_argument("--prior_buffer", type=int, default=512)
    parser.add_argument("--prior_min_samples", type=int, default=16)
    parser.add_argument("--prior_train_steps", type=int, default=2)
    parser.add_argument("--prior_height", type=int, default=160)
    parser.add_argument("--prior_width", type=int, default=480)
    parser.add_argument("--prior_ema", type=float, default=0.30)
    parser.add_argument("--prior_aug_noise", type=float, default=0.015)
    parser.add_argument("--prior_ckpt", type=str, default="")
    parser.add_argument(
        "--no_save_prior",
        action="store_true",
        help="Disable saving cnn_motion_prior.pt.",
    )

    parser.add_argument("--motion_smooth", type=float, default=0.25)

    # Particle Filter options
    parser.add_argument(
        "--no_particle_filter",
        action="store_true",
        help="Disable the 3D Particle Filter and save raw VO/CNN-prior position.",
    )
    parser.add_argument("--particle_count", type=int, default=1000)
    parser.add_argument("--particle_process_noise", type=float, default=0.035)
    parser.add_argument("--particle_measurement_noise", type=float, default=0.28)
    parser.add_argument("--particle_velocity_noise", type=float, default=0.08)
    parser.add_argument("--particle_resample_threshold", type=float, default=0.50)
    parser.add_argument("--particle_seed", type=int, default=7)

    # GPU acceleration options
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable available GPU acceleration for PyTorch CNN prior and OpenCV CUDA paths.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration even if CUDA is available.",
    )
    parser.add_argument(
        "--use_cuda_orb",
        action="store_true",
        help="Try OpenCV CUDA ORB feature extraction. Falls back to CPU ORB automatically.",
    )
    parser.add_argument(
        "--use_gpu_preprocess",
        action="store_true",
        help="Try OpenCV CUDA grayscale/CLAHE preprocessing. Falls back to CPU automatically.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed-precision CNN prior training/inference.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Try torch.compile on the CNN prior model for faster GPU execution.",
    )

    args = parser.parse_args()

    gpu_enabled = bool(args.gpu and not args.no_gpu)
    use_gpu_preprocess = bool((args.use_gpu_preprocess or gpu_enabled) and CUDA_CV_AVAILABLE)
    use_cuda_orb = bool((args.use_cuda_orb or gpu_enabled) and CUDA_ORB_AVAILABLE)
    use_amp = bool((not args.no_amp) and TORCH_AVAILABLE and torch.cuda.is_available())

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_dataset_files(data_dir)

    if files["video"] is None:
        print(f"[ERROR] No video found in {data_dir}")
        return 1

    cap = cv2.VideoCapture(str(files["video"]))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    except Exception:
        pass

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {files['video']}")
        return 1

    num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0 or not np.isfinite(fps):
        fps = 10.0

    ret, first_frame = cap.read()

    if not ret:
        print("[ERROR] Cannot read first video frame.")
        return 1

    height, width = first_frame.shape[:2]
    max_frames = num_frames_video if args.max_frames <= 0 else min(args.max_frames, num_frames_video)

    K = read_kitti_calibration(files["calib"])
    K = maybe_scale_intrinsics(
        K,
        video_width=width,
        video_height=height,
        original_width=args.orig_width,
        original_height=args.orig_height,
    )

    times = read_times(files["times"], max_frames, fps)
    _, gt_xyz = read_groundtruth(files["groundtruth"], times)

    if args.scale_mode == "gt" and gt_xyz is None:
        print("[ERROR] --scale_mode gt selected, but groundtruth.txt was not found.")
        return 1

    print("[INFO] Dataset:", data_dir)
    print("[INFO] Video:", files["video"])
    print("[INFO] Calibration:", files["calib"])
    print("[INFO] Ground truth:", files["groundtruth"])
    print("[INFO] Times:", files["times"])
    print(f"[INFO] Video size: {width}x{height}")
    print(f"[INFO] FPS: {fps:.3f}")
    print(f"[INFO] Frames: {max_frames}")
    print("[INFO] K:")
    print(K)
    print(f"[INFO] PyTorch CUDA available: {TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False}")
    print(f"[INFO] OpenCV CUDA available: {CUDA_CV_AVAILABLE}")
    print(f"[INFO] OpenCV CUDA ORB available: {CUDA_ORB_AVAILABLE}")
    print(f"[INFO] GPU requested: {gpu_enabled}")
    print(f"[INFO] GPU preprocessing enabled: {use_gpu_preprocess}")
    print(f"[INFO] CUDA ORB enabled: {use_cuda_orb}")
    print(f"[INFO] AMP enabled: {use_amp}")
    print(f"[INFO] Improved CNN prior input: {args.prior_width}x{args.prior_height}, train_steps/frame={args.prior_train_steps}, ema={args.prior_ema:.2f}")

    if args.use_cnn_prior and gt_xyz is None and not args.prior_ckpt:
        print(
            "[WARN] CNN prior enabled, but no groundtruth.txt or checkpoint was provided. "
            "The CNN will not learn metric scale. Temporal smoothing will still run."
        )

    frames_dir = outdir / "frames"

    if args.save_frames:
        frames_dir.mkdir(exist_ok=True)

    orb_cpu = cv2.ORB_create(
        nfeatures=args.nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        patchSize=31,
        fastThreshold=10,
    )

    orb_cuda = None
    if use_cuda_orb:
        try:
            orb_cuda = cv2.cuda_ORB.create(
                nfeatures=args.nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=19,
                patchSize=31,
                fastThreshold=10,
            )
        except Exception as e:
            print(f"[WARN] CUDA ORB creation failed; using CPU ORB: {e}")
            orb_cuda = None
            use_cuda_orb = False

    # Descriptor matching remains CPU-side for portability.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    prior_ckpt = Path(args.prior_ckpt) if args.prior_ckpt else None

    deep_prior = DeepPriorWrapper(
        enabled=args.use_cnn_prior,
        image_size=(args.prior_height, args.prior_width),
        lr=args.prior_lr,
        batch_size=args.prior_batch,
        buffer_size=args.prior_buffer,
        min_train_samples=args.prior_min_samples,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        ckpt=prior_ckpt,
        use_amp=use_amp,
        torch_compile_model=args.torch_compile,
        ema_alpha=args.prior_ema,
        aug_noise_std=args.prior_aug_noise,
    )

    smoother = MotionSmoother(alpha=args.motion_smooth)

    video_writer = None

    if not args.no_video:
        dash_w = width + args.map_size
        dash_h = max(height, args.map_size)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(outdir / "marked_vo_video.mp4"),
            fourcc,
            fps,
            (dash_w, dash_h),
        )

        if not video_writer.isOpened():
            print("[WARN] Video writer could not be opened.")
            video_writer = None

    R_w_c = np.eye(3, dtype=np.float64)

    # raw_t_w_c is the direct VO/CNN-prior accumulated translation.
    # t_w_c is the final filtered translation written to output files.
    raw_t_w_c = np.zeros(3, dtype=np.float64)
    t_w_c = np.zeros(3, dtype=np.float64)

    particle_filter = None
    if not args.no_particle_filter:
        particle_filter = ParticleFilter3D(
            n_particles=args.particle_count,
            init_pos=t_w_c,
            process_noise=args.particle_process_noise,
            measurement_noise=args.particle_measurement_noise,
            velocity_noise=args.particle_velocity_noise,
            resample_threshold=args.particle_resample_threshold,
            seed=args.particle_seed,
        )

    poses = []
    positions = []
    records = []

    prev_gray = preprocess_gray(first_frame, use_clahe=not args.no_clahe, use_cuda=use_gpu_preprocess)

    def append_pose(
        idx: int,
        stamp: float,
        base_scale: float,
        cnn_scale: Optional[float],
        cnn_conf: float,
        final_scale: float,
        cnn_alpha: float,
        info: Dict[str, object],
        raw_position: np.ndarray,
        particle_info: Dict[str, float],
    ):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_w_c
        T[:3, 3] = t_w_c

        qx, qy, qz, qw = rot_to_quat(R_w_c)

        poses.append(T.copy())
        positions.append(t_w_c.copy())

        raw_position = np.asarray(raw_position, dtype=np.float64).reshape(3)

        records.append(
            {
                "frame": idx,
                "time": float(stamp),
                "x": float(t_w_c[0]),
                "y": float(t_w_c[1]),
                "z": float(t_w_c[2]),
                "gt_x": float("nan"),
                "gt_y": float("nan"),
                "gt_z": float("nan"),
                "aligned_x": float("nan"),
                "aligned_y": float("nan"),
                "aligned_z": float("nan"),
                "error_x": float("nan"),
                "error_y": float("nan"),
                "error_z": float("nan"),
                "abs_error": float("nan"),
                "xz_error": float("nan"),
                "raw_x": float(raw_position[0]),
                "raw_y": float(raw_position[1]),
                "raw_z": float(raw_position[2]),
                "particle_enabled": int(particle_filter is not None),
                "particle_count": int(particle_info.get("particle_count", 0)),
                "particle_neff": float(particle_info.get("particle_neff", 0.0)),
                "particle_quality": float(particle_info.get("particle_quality", 0.0)),
                "particle_measurement_noise": float(
                    particle_info.get("particle_measurement_noise", 0.0)
                ),
                "particle_resampled": int(particle_info.get("particle_resampled", 0)),
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "base_scale": float(base_scale),
                "cnn_scale": float(cnn_scale) if cnn_scale is not None else 0.0,
                "cnn_conf": float(cnn_conf),
                "cnn_uncertainty": float(deep_prior.last_uncertainty) if np.isfinite(deep_prior.last_uncertainty) else 0.0,
                "cnn_pred_raw": float(deep_prior.last_pred_raw) if np.isfinite(deep_prior.last_pred_raw) else 0.0,
                "final_scale": float(final_scale),
                "cnn_alpha": float(cnn_alpha),
                "kp_prev": int(info.get("kp1", 0)),
                "kp_curr": int(info.get("kp2", 0)),
                "matches": int(info.get("matches", 0)),
                "inliers": int(info.get("inliers", 0)),
                "inlier_ratio": float(info.get("inlier_ratio", 0.0)),
                "gpu_orb": int(info.get("gpu_orb", 0)),
                "prior_loss": float(deep_prior.last_loss) if np.isfinite(deep_prior.last_loss) else 0.0,
            }
        )

    empty_info = {
        "kp1": 0,
        "kp2": 0,
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "gpu_orb": 0,
        "pts2": None,
    }

    append_pose(
        idx=0,
        stamp=times[0],
        base_scale=0.0,
        cnn_scale=None,
        cnn_conf=0.0,
        final_scale=0.0,
        cnn_alpha=0.0,
        info=empty_info,
        raw_position=raw_t_w_c,
        particle_info=particle_filter.info() if particle_filter is not None else {},
    )

    if args.save_frames:
        cv2.imwrite(str(frames_dir / "frame_000000.png"), first_frame)

    if video_writer is not None:
        online_error_stats = compute_online_error_stats(
            positions,
            gt_xyz,
            times,
            0,
        )

        map_img = draw_trajectory_map(
            positions,
            gt_xyz,
            0,
            args.map_size,
            args.map_scale,
            error_stats=online_error_stats,
        )

        overlay = first_frame.copy()

        cv2.putText(
            overlay,
            "Frame 0",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        dash = np.zeros((max(height, args.map_size), width + args.map_size, 3), dtype=np.uint8)
        dash[:height, :width] = overlay
        dash[:args.map_size, width:width + args.map_size] = map_img

        video_writer.write(dash)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    frame_idx = 0

    while frame_idx + 1 < max_frames:
        ret, curr_frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        if args.save_frames:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.png"), curr_frame)

        curr_gray = preprocess_gray(curr_frame, use_clahe=not args.no_clahe, use_cuda=use_gpu_preprocess)

        R_rel, t_rel, info = estimate_relative_pose_orb(
            prev_gray,
            curr_gray,
            K,
            orb_cpu,
            matcher,
            ransac_thresh=args.ransac_thresh,
            ratio=args.ratio,
            min_matches=args.min_matches,
            use_cuda_orb=use_cuda_orb,
            orb_cuda=orb_cuda,
        )

        base_scale = get_base_scale(
            frame_idx,
            gt_xyz,
            args.scale_mode,
            args.default_scale,
            args.min_scale,
            args.max_scale,
        )

        cnn_scale, cnn_conf = deep_prior.predict_scale(prev_gray, curr_gray)

        final_scale, cnn_alpha = blend_scale_with_cnn(
            base_scale=base_scale,
            cnn_scale=cnn_scale,
            cnn_conf=cnn_conf,
            prior_blend=args.prior_blend,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )

        dt_frame = float(times[frame_idx] - times[frame_idx - 1]) if frame_idx > 0 else 1.0 / max(fps, 1.0)
        if not np.isfinite(dt_frame) or dt_frame <= 0:
            dt_frame = 1.0 / max(fps, 1.0)

        particle_info = particle_filter.info() if particle_filter is not None else {}

        if R_rel is not None and t_rel is not None:
            R_old = R_w_c.copy()

            step_dir_world = R_old @ (-R_rel.T @ t_rel.reshape(3))

            norm_step = np.linalg.norm(step_dir_world)
            if norm_step > 1e-12:
                step_dir_world = step_dir_world / norm_step

            if args.invert_motion:
                step_dir_world = -step_dir_world

            measured_step = final_scale * step_dir_world

            filtered_step = smoother.filter(
                measured_step,
                inlier_ratio=float(info.get("inlier_ratio", 0.0)),
            )

            # Raw VO/CNN-prior position before particle filtering.
            raw_t_w_c = raw_t_w_c + filtered_step

            # Particle-filter reliability combines geometric inlier quality
            # and CNN-prior confidence. It is independent of ground truth.
            particle_quality = float(
                np.clip(
                    0.75 * float(info.get("inlier_ratio", 0.0))
                    + 0.25 * float(cnn_conf),
                    0.0,
                    1.0,
                )
            )

            if particle_filter is not None:
                t_w_c, particle_info = particle_filter.step(
                    measurement_pos=raw_t_w_c,
                    control_step=filtered_step,
                    dt=dt_frame,
                    quality=particle_quality,
                )
            else:
                t_w_c = raw_t_w_c.copy()

            R_w_c = R_old @ R_rel.T

        else:
            print(f"[WARN] Frame {frame_idx}: insufficient matches. Pose copied.")

            if particle_filter is not None:
                t_w_c, particle_info = particle_filter.step(
                    measurement_pos=None,
                    control_step=np.zeros(3, dtype=np.float64),
                    dt=dt_frame,
                    quality=0.0,
                )
            else:
                t_w_c = raw_t_w_c.copy()

        target_scale = gt_frame_scale(
            frame_idx,
            gt_xyz,
            args.min_scale,
            args.max_scale,
        )

        deep_prior.add_sample(prev_gray, curr_gray, target_scale)
        deep_prior.train_once(steps=args.prior_train_steps)

        append_pose(
            idx=frame_idx,
            stamp=times[frame_idx],
            base_scale=base_scale,
            cnn_scale=cnn_scale,
            cnn_conf=cnn_conf,
            final_scale=final_scale,
            cnn_alpha=cnn_alpha,
            info=info,
            raw_position=raw_t_w_c,
            particle_info=particle_info,
        )

        prev_gray = curr_gray

        if video_writer is not None:
            online_error_stats = compute_online_error_stats(
                positions,
                gt_xyz,
                times,
                frame_idx,
            )

            overlay = draw_feature_overlay(curr_frame, info.get("pts2", None))

            cv2.putText(
                overlay,
                f"Frame {frame_idx}/{max_frames - 1}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                f"matches={info.get('matches', 0)} inliers={info.get('inliers', 0)} "
                f"ratio={info.get('inlier_ratio', 0.0):.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                f"base={base_scale:.3f} cnn={cnn_scale if cnn_scale else 0:.3f} "
                f"conf={cnn_conf:.2f} unc={deep_prior.last_uncertainty if np.isfinite(deep_prior.last_uncertainty) else 0:.2f} "
                f"alpha={cnn_alpha:.2f} final={final_scale:.3f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                overlay,
                f"t=({t_w_c[0]:.2f}, {t_w_c[1]:.2f}, {t_w_c[2]:.2f}) "
                f"loss={deep_prior.last_loss if np.isfinite(deep_prior.last_loss) else 0:.4f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

            text_y = 175
            if particle_filter is not None:
                cv2.putText(
                    overlay,
                    f"PF: N={int(particle_info.get('particle_count', 0))} "
                    f"Neff={particle_info.get('particle_neff', 0.0):.0f} "
                    f"q={particle_info.get('particle_quality', 0.0):.2f} "
                    f"GPU_ORB={int(info.get('gpu_orb', 0))}",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                )
                text_y += 35

            if online_error_stats.get("available", False):
                cv2.putText(
                    overlay,
                    f"Online ATE RMSE={online_error_stats['running_rmse']:.3f} m "
                    f"x-z RMSE={online_error_stats['running_rmse_xz']:.3f} m "
                    f"curr={online_error_stats['current_error']:.3f} m",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )

            map_img = draw_trajectory_map(
                positions,
                gt_xyz,
                frame_idx,
                args.map_size,
                args.map_scale,
                error_stats=online_error_stats,
            )

            dash = np.zeros((max(height, args.map_size), width + args.map_size, 3), dtype=np.uint8)
            dash[:height, :width] = overlay
            dash[:args.map_size, width:width + args.map_size] = map_img

            video_writer.write(dash)

        if frame_idx % 50 == 0:
            print(
                f"[INFO] frame={frame_idx}/{max_frames - 1} "
                f"base_scale={base_scale:.4f} "
                f"cnn_scale={cnn_scale if cnn_scale else 0:.4f} "
                f"cnn_conf={cnn_conf:.2f} "
                f"cnn_unc={deep_prior.last_uncertainty if np.isfinite(deep_prior.last_uncertainty) else 0:.3f} "
                f"final_scale={final_scale:.4f} "
                f"pf_neff={particle_info.get('particle_neff', 0.0):.0f} "
                f"loss={deep_prior.last_loss if np.isfinite(deep_prior.last_loss) else 0:.5f}"
            )

    cap.release()

    if video_writer is not None:
        video_writer.release()

    if not args.no_save_prior and args.use_cnn_prior:
        deep_prior.save(outdir / "cnn_motion_prior.pt")

    vo_xyz = np.vstack(positions)

    error_report = compute_trajectory_error_report(vo_xyz, gt_xyz, times)
    fill_record_errors(records, error_report.get("error_rows", []))

    write_outputs(outdir, records, poses)

    save_trajectory_plot(
        outdir / "trajectory_xz.png",
        vo_xyz,
        gt_xyz,
        title="Monocular VO + Residual Attention Deep Prior + PF + GPU",
    )

    if error_report["available"]:
        n = int(error_report["n"])
        aligned = error_report["aligned"]
        sim_scale = float(error_report["sim_scale"])
        ate = float(error_report["ate_rmse"])

        save_trajectory_plot(
            outdir / "trajectory_aligned_xz.png",
            aligned,
            gt_xyz[:n],
            title=f"Aligned VO + Residual Attention Prior + PF + GPU, ATE RMSE = {ate:.4f}",
        )

        save_error_csv(
            outdir / "trajectory_error.csv",
            error_report["error_rows"],
        )

        save_error_plot(
            outdir / "error_over_time.png",
            error_report["error_rows"],
        )

        with open(outdir / "evaluation.txt", "w", encoding="utf-8") as f:
            f.write(f"frames_used {n}\n")
            f.write(f"similarity_scale {sim_scale:.9f}\n")
            f.write(f"ate_rmse {error_report['ate_rmse']:.9f}\n")
            f.write(f"ate_mae {error_report['ate_mae']:.9f}\n")
            f.write(f"ate_median {error_report['ate_median']:.9f}\n")
            f.write(f"ate_max {error_report['ate_max']:.9f}\n")
            f.write(f"ate_final {error_report['ate_final']:.9f}\n")
            f.write(f"rmse_x {error_report['rmse_x']:.9f}\n")
            f.write(f"rmse_y {error_report['rmse_y']:.9f}\n")
            f.write(f"rmse_z {error_report['rmse_z']:.9f}\n")
            f.write(f"rmse_xz {error_report['rmse_xz']:.9f}\n")
            f.write(f"cnn_prior_enabled {int(args.use_cnn_prior)}\n")
            f.write(f"cnn_train_steps {deep_prior.train_steps}\n")
            f.write(f"cnn_last_loss {deep_prior.last_loss}\n")
            f.write(f"cnn_last_uncertainty {deep_prior.last_uncertainty}\n")
            f.write(f"cnn_model ResidualAttentionCNNMotionPriorV2\n")
            f.write(f"prior_height {args.prior_height}\n")
            f.write(f"prior_width {args.prior_width}\n")
            f.write(f"prior_ema {args.prior_ema}\n")
            f.write(f"prior_aug_noise {args.prior_aug_noise}\n")
            f.write(f"motion_smooth {args.motion_smooth}\n")
            f.write(f"prior_blend {args.prior_blend}\n")
            f.write(f"particle_filter_enabled {int(particle_filter is not None)}\n")
            f.write(f"particle_count {args.particle_count}\n")
            f.write(f"particle_process_noise {args.particle_process_noise}\n")
            f.write(f"particle_measurement_noise {args.particle_measurement_noise}\n")
            f.write(f"particle_velocity_noise {args.particle_velocity_noise}\n")
            f.write(f"particle_resample_threshold {args.particle_resample_threshold}\n")
            f.write(f"gpu_requested {int(gpu_enabled)}\n")
            f.write(f"opencv_cuda_available {int(CUDA_CV_AVAILABLE)}\n")
            f.write(f"opencv_cuda_orb_available {int(CUDA_ORB_AVAILABLE)}\n")
            f.write(f"gpu_preprocess_enabled {int(use_gpu_preprocess)}\n")
            f.write(f"cuda_orb_enabled {int(use_cuda_orb)}\n")
            f.write(f"amp_enabled {int(use_amp)}\n")
            f.write(f"torch_compile {int(args.torch_compile)}\n")
            f.write("video_map_alignment display_only_umeyama_to_gt\n")
            f.write("video_map_grid removed\n")
            f.write("saved_plot_grid removed\n")
            f.write("video_map_online_rmse_panel enabled\n")
            f.write("video_overlay_online_rmse enabled\n")

        print(f"[RESULT] ATE RMSE: {error_report['ate_rmse']:.6f}")
        print(f"[RESULT] ATE MAE : {error_report['ate_mae']:.6f}")
        print(f"[RESULT] ATE MAX : {error_report['ate_max']:.6f}")
        print(f"[RESULT] RMSE x/y/z: {error_report['rmse_x']:.6f}, {error_report['rmse_y']:.6f}, {error_report['rmse_z']:.6f}")
        print(f"[RESULT] RMSE x-z: {error_report['rmse_xz']:.6f}")
    else:
        with open(outdir / "evaluation.txt", "w", encoding="utf-8") as f:
            f.write("error_metrics_available 0\n")
            f.write("reason groundtruth_not_available_or_too_short\n")
            f.write(f"cnn_prior_enabled {int(args.use_cnn_prior)}\n")
            f.write(f"particle_filter_enabled {int(particle_filter is not None)}\n")
            f.write(f"gpu_requested {int(gpu_enabled)}\n")
            f.write("video_map_online_rmse_panel enabled_but_waiting_for_groundtruth\n")

    print("[DONE] Outputs saved in:", outdir.resolve())
    print("       trajectory.csv")
    print("       trajectory_tum.txt")
    print("       trajectory_kitti.txt")
    print("       trajectory_xz.png")

    if gt_xyz is not None:
        print("       trajectory_aligned_xz.png")
        print("       trajectory_error.csv")
        print("       error_over_time.png")
        print("       evaluation.txt")

    if not args.no_video:
        print("       marked_vo_video.mp4")

    if args.use_cnn_prior and not args.no_save_prior:
        print("       cnn_motion_prior.pt")

    if particle_filter is not None:
        print("       particle_filter enabled in trajectory.csv")

    if gpu_enabled:
        print("       GPU acceleration requested; check [INFO] lines for active CUDA paths")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
