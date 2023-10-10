from fastapi import FastAPI, File, UploadFile;
from fastapi.middleware.cors import CORSMiddleware;
import tomato;
import common
import tensorflow as tf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/tomatoModel")
async def TomatoModel(
    file: UploadFile = File(...)
):
    image = common.read_file_as_image(await file.read())
    image_resized = tf.image.resize(image, size=[256, 256])
    payload = tomato.TomatoModel(image_resized)
    return payload

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}