import librosa
import sounddevice as sd
import numpy as np
import pandas as pd
from collections import deque

# -----------------------
# CONFIG
# -----------------------
AUDIO_FILE = "recordings/REC 03-25-2026 PALMA-12PM.wav"
WINDOW_DURATION = 5.0
STEP_DURATION = 5.0
SAMPLE_RATE = 16000
ROLLING_HISTORY = 3
OUTPUT_CSV = "palma-03-25-dataset12.csv"

NEAR_SILENCE_THRESHOLD = 0.05

def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# -----------------------
# FEATURE FUNCTION
# -----------------------
def extract_features(audio, sr):
    return {
        "avg_volume": np.mean(np.abs(audio)),
        "peak_volume": np.max(np.abs(audio)),
        "volume_variance": np.var(audio),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            audio, frame_length=len(audio), hop_length=len(audio)
        )[0, 0],
        "spectral_centroid": np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )
    }

# -----------------------
# LOAD AUDIO
# -----------------------
audio, sr = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE, mono=True)

window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
step_samples = int(STEP_DURATION * SAMPLE_RATE)

num_windows = (len(audio) - window_samples) // step_samples + 1

history = deque(maxlen=ROLLING_HISTORY)
rows = []

# previous values for deltas
prev = {
    "avg_volume": None,
    "peak_volume": None,
    "spectral_centroid": None,
    "zero_crossing_rate": None
}

quit_labeling = False

print("c = chatter | n = no chatter | q = stop labeling")

# -----------------------
# LOOP
# -----------------------
for i in range(num_windows):

    start = i * step_samples
    end = start + window_samples
    window = audio[start:end]

    if len(window) < window_samples:
        break

    # -----------------------
    # LABELING
    # -----------------------
    if not quit_labeling:
        sd.play(window, sr)
        sd.wait()

        while True:
            inp = input(f"[{format_time(start/sr)}-{format_time(end/sr)}] c/n/q: ")
            if inp in ["c", "n", "q"]:
                break

        if inp == "q":
            quit_labeling = True
            label = None
        else:
            label = 1 if inp == "c" else 0
    else:
        label = None

    # -----------------------
    # BASE FEATURES
    # -----------------------
    f = extract_features(window, sr)

    # -----------------------
    # DELTAS (ALWAYS DEFINED)
    # -----------------------
    f["volume_delta"] = 0 if prev["avg_volume"] is None else f["avg_volume"] - prev["avg_volume"]
    f["peak_delta"] = 0 if prev["peak_volume"] is None else f["peak_volume"] - prev["peak_volume"]
    f["centroid_delta"] = 0 if prev["spectral_centroid"] is None else f["spectral_centroid"] - prev["spectral_centroid"]
    f["zcr_decay"] = 0 if prev["zero_crossing_rate"] is None else f["zero_crossing_rate"] - prev["zero_crossing_rate"]

    prev["avg_volume"] = f["avg_volume"]
    prev["peak_volume"] = f["peak_volume"]
    prev["spectral_centroid"] = f["spectral_centroid"]
    prev["zero_crossing_rate"] = f["zero_crossing_rate"]

    # -----------------------
    # HISTORY FEATURES
    # -----------------------
    history.append(f)

    f["rolling_avg_volume"] = np.mean([x["avg_volume"] for x in history])
    f["rolling_peak_volume"] = np.mean([x["peak_volume"] for x in history])

    f["rolling_decay"] = (
        history[-1]["avg_volume"] - history[0]["avg_volume"]
        if len(history) > 1 else 0
    )

    f["near_silence"] = int(f["avg_volume"] < NEAR_SILENCE_THRESHOLD)

    f["centroid_volatility"] = (
        np.std([x["spectral_centroid"] for x in history])
        if len(history) > 1 else 0
    )

    # -----------------------
    # STREAK FEATURES (IMPORTANT FIX)
    # -----------------------
    if len(history) == 0:
        f["chatter_streak"] = 0
    else:
        if len(rows) > 0 and rows[-1].get("is_chatter") == 1:
            f["chatter_streak"] = rows[-1].get("chatter_streak", 0) + 1
        else:
            f["chatter_streak"] = 1 if label == 1 else 0

    f["streak_log"] = np.log1p(f["chatter_streak"])

    # -----------------------
    # METADATA
    # -----------------------
    f["timestamp_start"] = start / sr
    f["timestamp_end"] = end / sr
    f["is_chatter"] = label

    rows.append(f)

    del window

# -----------------------
# SAVE
# -----------------------
df = pd.DataFrame(rows)

# HARD GUARANTEE: all required columns exist
required_cols = [
    "chatter_streak",
    "streak_log",
    "volume_delta",
    "peak_delta",
    "centroid_delta",
    "zcr_decay",
    "rolling_decay",
    "near_silence",
    "centroid_volatility"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0

df.to_csv(OUTPUT_CSV, index=False)

print("Saved dataset:", OUTPUT_CSV)