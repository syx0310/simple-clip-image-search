from PIL import Image

import torch
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, ARRAY, ForeignKey, UniqueConstraint, PrimaryKeyConstraint,VARCHAR,select,desc
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

def query_database(cond, model, tokenizer, processor, device='cuda:0'):
	DATABASE_URL = config['db']['dburl']

	engine = create_engine(DATABASE_URL, echo=True)

	Base = declarative_base()

	# 定义数据模型
	class Test(Base):
		__tablename__ = 'test'

		id = mapped_column(VARCHAR, primary_key=True)
		embedding = mapped_column(Vector(768), nullable=False) # 对于向量我们使用数组类型
		embtype = mapped_column(Integer, primary_key=True)

		# 设置主键约束
		__table_args__ = (
			PrimaryKeyConstraint('id', 'embtype'),
		)

	class Md5Path(Base):
		__tablename__ = 'md5_path'

		id = mapped_column(Integer, primary_key=True)
		md5_hash = mapped_column(VARCHAR, nullable=False)
		file_path = mapped_column(VARCHAR, nullable=False)
	
	Session = sessionmaker(bind=engine)
	session = Session()
	
	with torch.no_grad():
		if isinstance(cond, str):
			input = tokenizer(cond, return_tensors="pt", padding=True).to(device)
			embeddings = model.get_text_features(**input)
		elif isinstance(cond, Image.Image):
			input = processor(images=cond, return_tensors="pt")
			embeddings = model.get_image_features(**input.to(device))

	# query = select(Test).order_by(Test.embedding.cosine_distance(embeddings[0])).limit(5)				
	out_dis = session.scalars(select(Test.embedding.cosine_distance(embeddings[0])).order_by(desc(Test.embedding.cosine_distance(embeddings[0])))).all()

	query = (
    select(Test, Md5Path.file_path)
    .join(Md5Path, Test.id == Md5Path.md5_hash)
    .order_by(Test.embedding.cosine_distance(embeddings[0]))
    .limit(5)
	)

	results = session.execute(query).all()

	# make a list of top5 file path
	out = []
	i = 0
	for test, file_path in results:
		out.append(file_path)
		i += 1
		if i == 5:
			break

	return [out, out_dis[0:5]]