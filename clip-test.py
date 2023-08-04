from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
inputs = tokenizer(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘", "花", "湖"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)

image_paths = ['./.assets/bird_image.jpg', './.assets/car_image.jpg']

images = [Image.open(img_path) for img_path in image_paths]

inputs = processor(images=images, return_tensors="pt")

image_features = model.get_image_features(**inputs)

print(torch.softmax(image_features@text_features.T, dim=-1))

print(text_features.shape)
print(image_features.shape)