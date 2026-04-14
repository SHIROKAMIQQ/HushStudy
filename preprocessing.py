import librosa
import numpy as np
import pandas as pd
from collections import deque

# =========================
# CONFIG
# =========================
AUDIO_FILE = "recordings/REC 03-25-2026 PALMA-12PM.wav"
SAMPLE_RATE = 48000
WINDOW_DURATION = 5  # seconds
STEP_DURATION = 5    # no overlap
ROLLING_HISTORY = 5  # for rolling features
OUTPUT_CSV = "datasets/OUTPUT PALMA-12pm.csv"

# =========================
# LOAD AUDIO
# =========================
audio, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE, mono=True)

window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
step_samples = int(STEP_DURATION * SAMPLE_RATE)

total_samples = len(audio)
num_windows = (total_samples - window_samples) // step_samples + 1

# -----------------------
# STORAGE
# -----------------------
history_buffer = deque(maxlen=ROLLING_HISTORY)
data_rows = []

print("Instructions:")
print("Enter 1 = chatter, 0 = no chatter, q = quit")
print("Then enter seconds until transition (chatter to not chatter, or vice versa)")

# -----------------------
# HELPERS
# -----------------------
def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------
# MAIN LOOP
# -----------------------
for i in range(num_windows):
    start_sample = i * window_samples
    end_sample = start_sample + window_samples

    window = audio[start_sample:end_sample]

    # -----------------------
    # TIME INFO
    # -----------------------
    start_time = start_sample / SAMPLE_RATE
    end_time = end_sample / SAMPLE_RATE

    print(f"\nClip {i+1}/{num_windows}")
    print(f"Time: {format_time(start_time)} - {format_time(end_time)}")

    # -----------------------
    # FEATURE EXTRACTION
    # -----------------------
    abs_window = np.abs(window)

    avg_volume = np.mean(abs_window)
    peak_volume = np.max(abs_window)
    volume_variance = np.var(abs_window)

    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(window))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))

    # -----------------------
    # ROLLING FEATURES
    # -----------------------
    history_buffer.append({
        "avg_volume": avg_volume,
        "peak_volume": peak_volume
    })

    rolling_avg_volume = np.mean([h["avg_volume"] for h in history_buffer])
    rolling_peak_volume = np.max([h["peak_volume"] for h in history_buffer])

    # -----------------------
    # LABEL INPUT
    # -----------------------
    while True:
        label = input()

        if label == "q":
            print("Stopping early...")
            break
        elif label in ["0", "1"]:
            label = int(label)
            break
        else:
            print("Invalid input. Enter 1, 0, or q.")
          
        chatter_duration = input()

    if label == "q":
        break

    # -----------------------
    # STORE ROW
    # -----------------------
    row = {
        "start_time": start_time,
        "end_time": end_time,
        "avg_volume": avg_volume,
        "peak_volume": peak_volume,
        "volume_variance": volume_variance,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid": spectral_centroid,
        "rolling_avg_volume": rolling_avg_volume,
        "rolling_peak_volume": rolling_peak_volume,
        "is_chatter": label,
        "chatter_duration": chatter_duration
    }

    data_rows.append(row)

# -----------------------
# SAVE CSV
# -----------------------
df = pd.DataFrame(data_rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved to {OUTPUT_CSV}")