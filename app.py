from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
from pathlib import Path

# Model loading
# NOTE: this path is produced by the pipeline "pusher" step.
DEFAULT_MODEL_PATH = Path("deployed_models") / "best.pt"

# Leave these as-is for now; you can tune later.
DEFAULT_CONF = 0.30
DEFAULT_IMGSZ = 640

model = None
if DEFAULT_MODEL_PATH.exists():
  model = YOLO(str(DEFAULT_MODEL_PATH))

app = FastAPI(title="Furniture Detector")

@app.get("/", response_class=HTMLResponse)
def home():
    warning = "" if model else "<p style=\"color:red;\">Model not found at deployed_models/best.pt. Train + push first.</p>"
    return f"""
    <html>
      <body style="font-family:Arial; text-align:center; margin-top:50px; background:#f0fff0;">
        <h1 style="color:green;">Furniture & Floorplan Detector LIVE</h1>
        {warning}
        <h2>Upload image</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input name="file" type="file" accept="image/*" required>
          <br><br>
          <button type="submit" style="padding:20px 60px; font-size:24px; background:#32CD32; color:white; border:none; border-radius:15px;">
            Detect Now
          </button>
        </form>
      </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  if model is None:
    return HTMLResponse("<h2 style='color:red;'>Model not available. Run training and pusher steps first.</h2>")
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    results = model(img, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ)[0]
    annotated = results.plot(line_width=3, font_size=1.5)
    _, buffer = cv2.imencode(".jpg", annotated)
    img64 = base64.b64encode(buffer).decode()

    detections = []
    for b in results.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        detections.append({
            "class": model.names[int(b.cls)],
            "confidence": round(float(b.conf), 3),
            "bbox": [x1, y1, x2, y2]
        })

    return HTMLResponse(f"""
    <center>
      <h1 style="color:green;">Found {len(detections)} Objects!</h1>
      <img src="data:image/jpeg;base64,{img64}" style="max-width:90%; border:8px solid lime; border-radius:20px;">
      <h2>JSON:</h2>
      <pre style="background:#e8ffe8; padding:20px; font-size:18px;">{json.dumps(detections, indent=2)}</pre>
      <a href="/" style="font-size:24px; color:green;">Upload Another</a>
    </center>
    """)