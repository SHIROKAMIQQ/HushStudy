import librosa
import sounddevice as sd
import numpy as np
import pandas as pd
from collections import deque
import math

WINDOW_DURATION = 5.0
STEP_DURATION = 5.0
SAMPLE_RATE = 16000
ROLLING_HISTORY = 3
NEAR_SILENCE_THRESHOLD = 0.05

def format_time(seconds):
  seconds = int(seconds)
  h = seconds // 3600
  m = (seconds % 3600) // 60
  s = seconds % 60
  return f"{h:02d}:{m:02d}:{s:02d}"

def extract_base_features(audio, sr):
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

def extract_features(input_path: str) -> pd.DataFrame:
  audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
  window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
  step_samples = int(STEP_DURATION * SAMPLE_RATE)
  num_windows = math.ceil(len(audio) / step_samples)

  history = deque(maxlen=ROLLING_HISTORY)
  rows = []

  prev = {
    "avg_volume": None,
    "peak_volume": None,
    "spectral_centroid": None,
    "zero_crossing_rate": None
  }
    
  for i in range(num_windows):
    start = i * step_samples
    end = start + window_samples
    window = audio[start:end]

    if len(window) == 0:
      break
      
    f = extract_base_features(window, sr)
    f["volume_delta"] = 0 if prev["avg_volume"] is None else f["avg_volume"] - prev["avg_volume"]
    f["peak_delta"] = 0 if prev["peak_volume"] is None else f["peak_volume"] - prev["peak_volume"]
    f["centroid_delta"] = 0 if prev["spectral_centroid"] is None else f["spectral_centroid"] - prev["spectral_centroid"]
    f["zcr_decay"] = 0 if prev["zero_crossing_rate"] is None else f["zero_crossing_rate"] - prev["zero_crossing_rate"]

    prev["avg_volume"] = f["avg_volume"]
    prev["peak_volume"] = f["peak_volume"]
    prev["spectral_centroid"] = f["spectral_centroid"]
    prev["zero_crossing_rate"] = f["zero_crossing_rate"]

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

    rows.append(f)
    del window

  return pd.DataFrame(rows)
        
  
    


