from fastapi import FastAPI, UploadFile, File, Form, Request, Response, Cookie, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import shutil
import subprocess
import os
import uuid # uuid to be used for cookies

import pandas as pd
import chatter_classifier
import duration_prediction
import feature_extraction

import sqlite3
import time

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_STUDYABLE_THRESHOLD = 60 * 30 # in seconds
LEARNING_RATE = 0.3 #arbitrary 

# initialize DB
conn = sqlite3.connect("thresholdprefs.db")
conn.execute("""
  CREATE TABLE IF NOT EXISTS thresholds (
    user_id TEXT PRIMARY KEY,
    threshold INTEGER NOT NULL DEFAULT 1800
  )
""")
conn.close()

#helper function for DB access
def get_threshold(user_id: str) -> int:
  with sqlite3.connect("thresholdprefs.db") as conn:
    cur = conn.execute("SELECT threshold FROM thresholds WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if row:
      return row[0]
    else:
      conn.execute("INSERT INTO thresholds (user_id, threshold) VALUES (?, ?)", (user_id, DEFAULT_STUDYABLE_THRESHOLD))
      return DEFAULT_STUDYABLE_THRESHOLD

def set_threshold(user_id: str, new_threshold: int):
  with sqlite3.connect("thresholdprefs.db") as conn:
    conn.execute(
      """
      INSERT INTO thresholds (user_id, threshold)
      VALUES (?, ?)
      ON CONFLICT(user_id)
      DO UPDATE SET threshold = excluded.threshold
      """,
      (user_id, new_threshold)
    )

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

#helper function for cookie gen
def get_user_id(request: Request, response: Response) -> str:
  user_id = request.cookies.get("user_id")
  if not user_id:
    user_id = str(uuid.uuid4())
    response.set_cookie(
      key="user_id",
      value=user_id,
      max_age = 31536000, #1 yr in sec
      httponly = True,
      secure = False,
      samesite="lax",
    )
    set_threshold(user_id, DEFAULT_STUDYABLE_THRESHOLD)
  return user_id

#endpoint for model adjustment to threshold
"""
Given a duration prediction:, 
  Use arbitrary learning rate to adjust the 
  user's saved studyable threshold and update
  database accordingly.
"""
@app.post("/adjust-threshold")
async def adjust_threshold(duration_prediction: int, request: Request = None, response: Response = None):
  user_id = get_user_id(request, response)
  old_threshold = get_threshold(user_id)
  new_threshold = int( (1-LEARNING_RATE) * old_threshold + (LEARNING_RATE) * duration_prediction )
  set_threshold(user_id, new_threshold)
  return {"message": "Threshold adjusted", "new_threshold": new_threshold}

"""
Given an input audio file tmp.webm:
  We want to temporarily save it as a .wav file.
  Preprocess it, with is_chatter and chatter_streak blank.
  Let the models predict is_chatter and chatter_streak values.
  Then output the result based on the last audio clip.

  Modifications to incorporate designated user threshold
"""
@app.post("/upload-audio")
async def upload_audio(request: Request, response: Response, file: UploadFile = File(...)):

  total_start = time.perf_counter()

  user_id = get_user_id(request, response)
  print(user_id)
  user_threshold = get_threshold(user_id)
    
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

  studyable = False

  if last_is_chatter:
    y_duration_prediction = duration_prediction.model.predict(X[duration_prediction.feature_cols])
    if int(y_duration_prediction[-1]) <= user_threshold:
      studyable = True
  else:
    y_duration_prediction = [None] * len(y_is_chatter)
    studyable = True

  
  total_end = time.perf_counter()
  total_latency = total_end - total_start
  print(f"LATENCY ASSESSMENT: {total_latency}")

  debug_df = pd.DataFrame({
    "is_chatter": y_is_chatter,
    "duration_prediction": y_duration_prediction,
    "studyable": studyable
  })
  print(debug_df)

  output = JSONResponse({
    "is_chatter": last_is_chatter,
    "duration_seconds_left": int(y_duration_prediction[-1]) if y_duration_prediction[-1] != None else None,
    "studyable": studyable
  })
  output.set_cookie(
    key="user_id",
    value=user_id,
    max_age = 31536000, #1 yr in sec
    httponly = True,
    secure = False,
    samesite="lax",
  )
  print(output)
  return output 
  
