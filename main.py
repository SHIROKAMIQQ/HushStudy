from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from pathlib import Path
import shutil
import subprocess
import os

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", "."))
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
  
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

  return {
    "message": "converted",
    "wav_path": str(output_path)
  }
  
