"""
Tracking videosundan kisa (birkac saniyelik), kucuk boyutlu bir GIF
onizlemesi uretir -- boylece GitHub README icinde otomatik oynar.
"""
import cv2
from PIL import Image

VIDEO_PATH = r"results\videos\v_00HRwkvvjtQ_c001_bytetrack.mp4"
OUTPUT_GIF = r"assets\tracking_preview.gif"
MAX_FRAMES = 90       # ~3-4 saniyelik onizleme (fps'e gore degisir)
EVERY_NTH = 2          # her 2 karede bir al (dosya boyutunu kucultur)
RESIZE_WIDTH = 480     # GIF genisligi (px)

cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
count = 0

while cap.isOpened() and len(frames) < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    if count % EVERY_NTH == 0:
        h, w = frame.shape[:2]
        new_h = int(h * (RESIZE_WIDTH / w))
        frame = cv2.resize(frame, (RESIZE_WIDTH, new_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    count += 1

cap.release()

frames[0].save(
    OUTPUT_GIF, save_all=True, append_images=frames[1:],
    duration=80, loop=0, optimize=True
)
print(f"Kaydedildi: {OUTPUT_GIF} ({len(frames)} kare)")