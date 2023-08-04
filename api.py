from PIL import Image
import requests
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer

from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import io

app = FastAPI()

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")

@app.post("/embimage")
async def process_image(file: UploadFile = File(...)):
	contents = await file.read()
	image = Image.open(io.BytesIO(contents))
	inputs = processor(images=[image], return_tensors="pt")
	image_features = model.get_image_features(**inputs)
	return {'feature': image_features.tolist()[0]}

@app.post("/embtext")
async def process_text(text: str = Form(...)):
	inputs = tokenizer([text], padding=True, return_tensors="pt")
	text_features = model.get_text_features(**inputs)
	return {'feature': text_features.tolist()[0]}

