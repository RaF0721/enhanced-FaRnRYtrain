from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os

model = YOLO("Path_to_your_model")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(..., max_size=50_000_000)): 
    with open("load.jpg", "wb") as buffer:
        buffer.write(await file.read())

    results = model("load.jpg")

    annotated_image_path = "annotated_image.jpg"
    results[0].save(annotatd_image_path)

    return FileResponse(annotated_image_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)