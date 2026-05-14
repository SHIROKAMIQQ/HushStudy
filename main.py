from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import shutil
import subprocess
import os

import pandas as pd
import chatter_classifier
import duration_prediction
import feature_extraction

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

STUDYABLE_THRESHOLD = 60 * 30 # in seconds

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

"""
Given an input audio file tmp.webm:
  We want to temporarily save it as a .wav file.
  Preprocess it, with is_chatter and chatter_streak blank.
  Let the models predict is_chatter and chatter_streak values.
  Then output the result based on the last audio clip.
"""
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):

  # Save input audio as tmp/tmp.wav  
  input_path = TMP_DIR / "tmp.webm"
  output_path = TMP_DIR / "tmp.wav"

  with open(input_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)

  subprocess.run([
    "ffmpeg",
    "-y",
    "-i", input_path,
    "-ar", "48000",
    "-ac", "1",
    output_path
  ], check=True)

  os.remove(input_path)

  # Preprocess tmp.wav into features df
  X = feature_extraction.extract_features("tmp/tmp.wav")
  print(X.columns)
  print(X.shape)

  # Predict using models
  X_scaled = chatter_classifier.scaler.transform(X[chatter_classifier.feature_cols])
  y_is_chatter = chatter_classifier.model.predict(X_scaled)
  last_is_chatter = bool(y_is_chatter[-1])

  if last_is_chatter:
    y_duration_prediction = duration_prediction.model.predict(X[duration_prediction.feature_cols])
  else:
    y_duration_prediction = [None] * len(y_is_chatter)

  debug_df = pd.DataFrame({
    "is_chatter": y_is_chatter,
    "duration_prediction": y_duration_prediction
  })
  print(debug_df)

  output = {
    "is_chatter": last_is_chatter,
    "duration_left_seconds": int(y_duration_prediction[-1]) if y_duration_prediction[-1] != None else None
  }
  print(output)
  return output 
  
