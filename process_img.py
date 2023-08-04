from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, ARRAY, ForeignKey, UniqueConstraint, PrimaryKeyConstraint,VARCHAR
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from get_file_path import get_file_path
from all_md5 import get_all_md5

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

DATABASE_URL = config['db']['dburl']

engine = create_engine(DATABASE_URL, echo=True)

Base = declarative_base()

# 定义数据模型
class Test(Base):
    __tablename__ = 'test'

    id = mapped_column(VARCHAR, primary_key=True)
    embedding = mapped_column(Vector(768), nullable=False) 
    embtype = mapped_column(Integer, primary_key=True)

    # 设置主键约束
    __table_args__ = (
        PrimaryKeyConstraint('id', 'embtype'),
    )
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px").to(device)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
# tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")

path = r'./.assets'
file_list = []
dir_list = []
get_file_path(path, file_list, dir_list)
md5_list = get_all_md5(file_list)

split_size = 4
file_list = [file_list[i:i + split_size] for i in range(0, len(file_list), split_size)]
md5_list = [md5_list[i:i + split_size] for i in range(0, len(md5_list), split_size)]

Session = sessionmaker(bind=engine)
session = Session()

with torch.no_grad():
	st = time.time()
	for i in range(len(file_list)):
		images = [Image.open(img_path) for img_path in file_list[i]]
		inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
		image_features = model.get_image_features(**inputs)
		for j in range(len(file_list[i])):
			test = Test(id=md5_list[i][j], embedding=image_features[j].tolist(), embtype=1)
			session.add(test)
			session.commit()
	print(time.time() - st)
