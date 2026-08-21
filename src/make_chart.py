"""README icin karsilastirma grafigini uretir."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

trackers = ["SORT", "ByteTrack", "OC-SORT", "BoT-SORT", "C-BIoU"]
mota = [69.165, 74.938, 67.151, 73.519, 73.865]
hota = [44.554, 47.542, 43.829, 44.165, 45.438]
idf1 = [41.896, 49.184, 44.500, 43.726, 47.266]

x = np.arange(len(trackers))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.bar(x - width, mota, width, label="MOTA", color="#2a78d6")
ax.bar(x, hota, width, label="HOTA", color="#eb6834")
ax.bar(x + width, idf1, width, label="IDF1", color="#1baf7a")

ax.set_ylabel("Skor (%)")
ax.set_title("RT-DETR + Tracker Karsilastirmasi -- SportsMOT")
ax.set_xticks(x)
ax.set_xticklabels(trackers)
ax.legend()
ax.set_ylim(0, 85)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("assets/tracker_comparison.png")
print("Kaydedildi: assets/tracker_comparison.png")