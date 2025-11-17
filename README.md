# 🎞️ AI Video Frame Interpolation — Powered by RIFE

Welcome to the **AI Frame Interpolation System** — a tool designed to generate ultra-smooth videos by increasing framerate using the **RIFE (Real-Time Intermediate Flow Estimation)** deep-learning model.

This project allows you to turn low-FPS video into high-FPS cinematic content by generating high-quality intermediate frames with GPU acceleration.

---

## 🚀 Features

- ⚡ **High-speed frame interpolation with GPU**
- 🎬 Convert **24 → 48 → 96 FPS**, **30 → 60 → 120 FPS**, **25 → 50 FPS**, etc.
- 🎯 Set *exact FPS outputs* or use exponential interpolation
- 📂 Simple input/output folder workflow
- 🎥 FFmpeg-powered video reading + encoding
- 🧠 Uses RIFE model for high-accuracy temporal prediction
- 🛠️ Supports UHD mode for high-resolution videos
- 🧹 Duplicate-frame skipping available

---

## 📦 Requirements

Before running, make sure you have:

- **Python 3.8+**
- **NVIDIA GPU recommended**
- **PyTorch (CUDA)** installed  
- **FFmpeg** installed and added to PATH  
- **RIFE model files** placed inside the project directory  
- Your input video stored at:  
