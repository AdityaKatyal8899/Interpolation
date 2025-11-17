import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import skvideo.io
from queue import Queue
from model.pytorch_msssim import ssim_matlab
import subprocess, shutil, time

# ============================================================
# 🧩 NumPy Compatibility (for modern versions)
# ============================================================
np.float = float
np.int = int

_old_fromstring = getattr(np, "fromstring", None)
def _fromstring_compat(data, *args, **kwargs):
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, *args, **kwargs)
    try:
        return _old_fromstring(data, *args, **kwargs)
    except ValueError as e:
        if "binary mode" in str(e).lower():
            return np.frombuffer(data, *args, **kwargs)
        raise e
np.fromstring = _fromstring_compat

warnings.filterwarnings("ignore")


# ============================================================
# 🎵 Audio Transfer (Pure FFmpeg – No MoviePy)
# ============================================================
def transferAudio(sourceVideo, targetVideo):
    temp_dir = "./temp"
    temp_audio = os.path.join(temp_dir, "audio.m4a")

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Extract audio
    subprocess.call([
        "ffmpeg", "-y", "-i", sourceVideo,
        "-vn", "-acodec", "copy", temp_audio
    ])

    # Remove audio from interpolated video
    target_noaudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    if os.path.exists(targetVideo):
        os.rename(targetVideo, target_noaudio)

    # Merge audio into final output
    subprocess.call([
        "ffmpeg", "-y",
        "-i", target_noaudio,
        "-i", temp_audio,
        "-c", "copy", targetVideo
    ])

    if os.path.exists(target_noaudio):
        os.remove(target_noaudio)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# ⚙️ Argument Parser
# ============================================================
parser = argparse.ArgumentParser(description='RIFE Frame Interpolation')
parser.add_argument('--video', type=str, required=True, help='Input video path')
parser.add_argument('--output', type=str, default=None, help='Output video path')
parser.add_argument('--model', type=str, default='train_log', help='Trained model directory')
parser.add_argument('--fp16', action='store_true', help='Enable half precision (Tensor Cores)')
parser.add_argument('--UHD', action='store_true', help='Support 4K')
parser.add_argument('--scale', type=float, default=1.0, help='Scale factor (use 0.5 for 4K)')
parser.add_argument('--fps', type=int, default=None, help='Force output FPS')
parser.add_argument('--exp', type=int, default=1, help='Interpolation exponent (2^exp = FPS multiplier)')
args = parser.parse_args()


# ============================================================
# 🚀 Device Setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 CUDA version: {torch.version.cuda}")
else:
    print("⚠️ GPU not found. Running on CPU — much slower.")
print(f"{'='*60}\n")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)


# ============================================================
# 📦 Load Model (Auto-detect version)
# ============================================================
try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.model, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.model, -1)
        print("Loaded v3.x HD model.")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.model, -1)
    print("Loaded base RIFE model.")

model.eval()

# ✅ Proper device transfer for RIFE's internal networks
if hasattr(model, 'device'):
    try:
        model.device()  # some RIFE versions use this custom method
    except:
        pass
# Move internal networks (these exist in all versions)
if hasattr(model, 'flownet'):
    model.flownet.to(device)
if hasattr(model, 'contextnet'):
    model.contextnet.to(device)
if hasattr(model, 'fusionnet'):
    model.fusionnet.to(device)

print(f"✅ Model moved to: {device}\n")



# ============================================================
# 🎥 Video Setup
# ============================================================
video_path = args.video
videoCapture = cv2.VideoCapture(video_path)
fps_in = videoCapture.get(cv2.CAP_PROP_FPS)
total_frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
videoCapture.release()

fps_out = args.fps if args.fps else fps_in * (2 ** args.exp)
fpsNotAssigned = args.fps is None

print(f"Input: {fps_in:.2f} FPS | Frames: {total_frames}")
print(f"Output target: {fps_out:.2f} FPS | Exp={args.exp}\n")

videogen = skvideo.io.vreader(video_path)
lastframe = next(videogen)
h, w, _ = lastframe.shape

output_name = args.output or f"{os.path.splitext(video_path)[0]}_{2**args.exp}X_{int(fps_out)}fps.mp4"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_out = cv2.VideoWriter(output_name, fourcc, fps_out, (w, h))


# ============================================================
# 🧠 Helper Functions
# ============================================================
def make_inference(I0, I1, n):
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first = make_inference(I0, middle, n=n // 2)
    second = make_inference(middle, I1, n=n // 2)
    return [*first, middle, *second] if n % 2 else [*first, *second]


def pad_image(img, padding):
    return F.pad(img, padding).half() if args.fp16 else F.pad(img, padding)


# ============================================================
# 🧩 Core Interpolation Loop
# ============================================================
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

pbar = tqdm(total=total_frames)
I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1, padding)
temp = None

for frame in videogen:
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, padding)

    output_frames = make_inference(I0, I1, (2 ** args.exp) - 1)
    vid_out.write((I0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w][:, :, ::-1])
    for mid in output_frames:
        mid = ((mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w]
        vid_out.write(mid[:, :, ::-1])

    pbar.update(1)

pbar.close()
vid_out.release()

# ============================================================
# 🔊 Audio Merge
# ============================================================
if fpsNotAssigned:
    try:
        transferAudio(video_path, output_name)
    except Exception as e:
        print(f"Audio merge failed: {e}")
else:
    print("Skipping audio merge (custom FPS set).")

print(f"\n✅ Done! Output saved as: {output_name}")
print("Check GPU usage with: nvidia-smi\n")
