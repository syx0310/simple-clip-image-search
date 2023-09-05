from PIL import Image
from io import BytesIO
from sqlalchemy import create_engine, Column, Integer, String, PrimaryKeyConstraint,VARCHAR, insert, select,desc,text

from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from transformers import ChineseCLIPProcessor,ChineseCLIPModel
import torch
from loguru import logger
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Body
from typing import List
from pydantic import BaseModel
import os
from time import sleep
import psutil
import sys

api = FastAPI()
torch.manual_seed(0)

def load_conf_from_env():
	# Load config from environment variables
	conf = {}
	conf['dburl'] = os.environ.get('DBURL', 'postgresql://mtphotos:mtphotos@clip-db:5432/mtphotos')
	conf['model'] = os.environ.get('MODEL', 'OFA-Sys/chinese-clip-vit-base-patch16')
	conf['model_path'] = os.environ.get('MODEL_PATH', '/data/model')
	conf['device'] = os.environ.get('DEVICE', 'cpu')
	conf['clip_api_key'] = os.environ.get('CLIP_API_KEY', 'YOUR_SECRET_API_KEY')
	conf['log_level'] = os.environ.get('LOG_LEVEL', 'INFO')
	return conf

conf = load_conf_from_env()
logger.add("/data/main.log", rotation="100 MB", level=conf['log_level'])

if conf['model'] == 'OFA-Sys/chinese-clip-vit-base-patch16':
	EMB_DIM = 512
else:
	EMB_DIM = 768

Base = declarative_base()

class embimg(Base):
	__tablename__ = 'embimg'

	md5 = mapped_column(VARCHAR, primary_key=True)
	embedding = mapped_column(Vector(EMB_DIM), nullable=True)

	__table_args__ = (
		PrimaryKeyConstraint('md5'),
	)

class embtext(Base):
    __tablename__ = 'embtext'

    id = Column(Integer, primary_key=True, autoincrement=True) # Auto-incrementing ID
    contents = Column(String, nullable=False) # Text contents
    embedding = Column(Vector(768), nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint('id'),
    )

class SearchRequest(BaseModel):
    text: str
    start: int
    end: int

class DeleteImagesRequest(BaseModel):
    md5s: List[str]

class RestartModel(BaseModel):
    model: str

DATABASE_URL = conf['dburl']
# engine = create_engine(DATABASE_URL, echo=True, pool_recycle=3600, pool_pre_ping=True)
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_pre_ping=True)
session = sessionmaker(bind=engine)()

# # check db
# try:
# 	# session = make_session()
# 	session.query(embimg)
# 	logger.info('db connected')
# except Exception as e:
# 	logger.error('db error:{}'.format(e))
# 	exit()

model = None
processor = None

def get_api_key(
    api_key: str = Header(None, alias="api_key"),
):
    if api_key != conf['clip_api_key']:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

def load_model():
	global processor, model
	try:
		processor = ChineseCLIPProcessor.from_pretrained(conf['model'], cache_dir=conf['model_path'])
		model = ChineseCLIPModel.from_pretrained(conf['model'], cache_dir=conf['model_path'])
	except Exception as e:
		logger.error('model load error:{}'.format(e))

def restart_server():
	logger.info('restarting server...')

	python = sys.executable
	os.execl(python, python, *sys.argv)

async def write_imgdb(md5, embedding):
	try:
		record = embimg(md5=md5, embedding=embedding)
		session.merge(record)
		session.commit()
	except Exception as e:
		logger.error('db error:{}'.format(e))
		session.rollback()
		raise HTTPException(status_code=500, detail=str(e))

async def write_textdb(contents, embedding):
	try:
		existing_record = session.query(embtext).filter_by(contents=contents).first()
		
		if existing_record:
			# If the content exists, update the embedding
			existing_record.embedding = embedding
		else:
			# If the content does not exist, create a new record
			record = embtext(contents=contents, embedding=embedding)
			session.add(record)
		# Commit the transaction
		session.commit()
	except Exception as e:
		logger.error('db error:{}'.format(e))
		session.rollback()
		raise HTTPException(status_code=500, detail=str(e))

async def searchDB(session, model, processor, tokenizer, cond):
	device = model.device
	if isinstance(cond, str):
		existing_embedding_query = select(embtext.embedding).filter(embtext.contents == cond)
		existing_embedding = session.execute(existing_embedding_query).scalar_one_or_none()
		embeddings = existing_embedding
	else:
		existing_embedding = None

	if existing_embedding is None:
		with torch.no_grad():
			if isinstance(cond, str):
				input = tokenizer(cond, return_tensors="pt", padding=True).to(device)
				embeddings = model.get_text_features(**input)
			elif isinstance(cond, Image.Image):
				input = processor(images=cond, return_tensors="pt").to(device)
				embeddings = model.get_image_features(**input)
	if embeddings[0].shape[0] != EMB_DIM:
		logger.error(f"Embedding dimension mismatch, please check your model.")
		raise HTTPException(status_code=500, detail="Embedding dimension mismatch, please check your model.")
	query = (
		select(embimg.md5)
		.where(embimg.embedding.isnot(None))
		.order_by(embimg.embedding.cosine_distance(embeddings[0]))
	)
	result = session.execute(query).all()
	out_dis = session.scalars(
		select(embimg.embedding.cosine_distance(embeddings[0]))
		.where(embimg.embedding.isnot(None))
		.order_by(desc(embimg.embedding.cosine_distance(embeddings[0])))).all()
	logger.info(f"Search result {len(result)}")
	# logger.info(f"Search result {len(out_dis)}")
	return result, out_dis


@api.get("/api/initdb", dependencies=[Depends(get_api_key)])
async def initdb():
	init_sqls = ['DROP EXTENSION IF EXISTS vector;',
			  'CREATE EXTENSION vector;',
				'DROP TABLE IF EXISTS embimg;',
			  f'CREATE TABLE embimg (md5 VARCHAR ,embedding VECTOR({EMB_DIM}), CONSTRAINT pk_embimg PRIMARY KEY (md5));',
			  'DROP TABLE IF EXISTS embtext;',
			  f'CREATE TABLE embtext (id SERIAL ,contents VARCHAR,embedding VECTOR({EMB_DIM}), CONSTRAINT pk_embtext PRIMARY KEY (id));']

	try:
		for sql in init_sqls:
			session.execute(text(sql))
		session.commit()
		return {'status': 'done'}
	except Exception as e:
		logger.error('db error:{}'.format(e))
		session.rollback()
		raise HTTPException(status_code=500, detail=str(e))


@api.post("/api/embimage")
async def process_images(md5s: List[str], files: List[UploadFile] = File(...)):
	if model is None:
		# return HTTPException(status_code=500, detail="Vision model not loaded")
		return {"status": f"Error: vision model not loaded", 'count': 0 }
	device = model.device
	try:
		images = [Image.open(BytesIO(file.file.read())) for file in files]
	except Exception as e:
		logger.error(f"Error: file open related error occurred: {str(e)}")
		return {"status": f"Error: file open related error occurred: {str(e)}", 'count': 0 }
	try: 
		inputs = processor(images=images, return_tensors="pt")
		with torch.no_grad():
			if device != 'cpu':
				inputs = inputs.to(device)
			image_features = model.get_image_features(**inputs)
		for md5, embedding in zip(md5s, image_features.tolist()):
			try:
				await write_imgdb(md5, embedding)
			except Exception as e:
				logger.error(f"Error: db error occurred: {str(e)}")
				return {"status": f"Error: db error occurred: {str(e)}", 'count': 0 }
		logger.info(f"Image {md5s} processed successfully.")
		logger.debug('ram usage: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
		return {'status': 'done', 'count': len(image_features.tolist())}
	except RuntimeError as e:
		if "out of memory" in str(e):
			logger.error(f"Error: GPU memory overflow. Please try with smaller batch size.")
			return {"status": "Error: GPU memory overflow. Please try with smaller batch size.", 'count': 0 }
		else:
			logger.error(f"Error: unexpected error occurred: {str(e)}")
			return {"status": f"Error: unexpected error occurred: {str(e)}", 'count': 0 }

@api.post("/api/embtext", dependencies=[Depends(get_api_key)])
async def process_text(text: List[str]):
	if model is None:
		# return HTTPException(status_code=500, detail="Text model not loaded")
		return {"status": f"Error: text model not loaded", 'count': 0 }
	device = model.device
	try:
		inputs = processor(text, padding=True, return_tensors="pt")
		with torch.no_grad():
			if device != 'cpu':
				inputs = inputs.to(device)
			text_features = model.get_text_features(**inputs)
		for contents, embedding in zip(text, text_features.tolist()):
			await write_textdb(contents, embedding)
		return {'status': 'done', 'count': len(text_features.tolist())}
	except RuntimeError as e:
		if "out of memory" in str(e):
			return {"status": "Error: GPU memory overflow. Please try with shorter texts or fewer texts.", 'count': 0 }
		else:
			return {"status": f"Error: unexpected error occurred: {str(e)}", 'count': 0 }

@api.post("/api/search", dependencies=[Depends(get_api_key)])
async def search(request: SearchRequest):
	if model is None:
		# return HTTPException(status_code=500, detail="Text model not loaded")
		return {"status": f"Error: text model not loaded", 'count': 0 }
	text = request.text
	start = request.start
	end = request.end
	if end < start and end > 0:
		end = start + 1
	if start < 0:
		start = 0
	result, out_dis = await searchDB(session, model, processor, processor, text)
	if end < 0 or end > len(result):
		end = len(result)
	# print(result)
	# [('48248', '13f5705af279f1c5fd24a0a37d02dc60')]
	md5 = [i[0] for i in result[start:end]]
	return {'md5': md5, 'dis': out_dis[start:end]}

@api.get("/api/cuda", dependencies=[Depends(get_api_key)])
async def get_cuda():
	return {'cuda': torch.cuda.is_available()}

@api.get("/api/device", dependencies=[Depends(get_api_key)])
async def get_device():
	if str(model.device)!='cpu':
		return {'device': str(model.device), 'totalvram': torch.cuda.get_device_properties(0).total_memory}
	else:
		return {'device': 'cpu', 'totalvram': 0}
	
@api.post("/api/switchmodel", dependencies=[Depends(get_api_key)])
def switchmodel(request_model:RestartModel):

    restart_server()

    return {"message": f"Restarting with {request_model.model} model"}

@api.get("/api/move2cpu", dependencies=[Depends(get_api_key)])
def move2cpu():
	try:
		model.to('cpu')
		return {'status': 'done','device': str(model.device)}
	except Exception as e:
		logger.error('torch error:{}'.format(e))
		raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/move2gpu", dependencies=[Depends(get_api_key)])
def move2gpu():
	try: 
		if torch.cuda.is_available():
			model.to(conf['device'])
			return {'status': 'done','device': str(model.device)}
		else:
			return {'status': 'error', 'device': 'cpu'}
	except Exception as e:
		logger.error('cuda error:{}'.format(e))
		raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/getallimagemd5", dependencies=[Depends(get_api_key)])
async def getAllImageMD5():
	query = (
		select(embimg.md5)
		.where(embimg.embedding.isnot(None))
	)
	result = session.execute(query).all()
	md5 = [i[0] for i in result]
	return {'md5': md5}

@api.get("/api/getAllTextInfo", dependencies=[Depends(get_api_key)])
async def getAllTextInfo():
	query = (
		select(embtext.id, embtext.contents)
		.where(embtext.embedding.isnot(None))
	)
	id = []
	contents = []
	for i in session.execute(query).all():
		id.append(i[0])
		contents.append(i[1])

	return {'id': id, 'contents': contents}

@api.delete("/api/deletetext", dependencies=[Depends(get_api_key)])
async def delete_text(contents: str):
	try:
		record = session.query(embtext).filter(embtext.contents == contents).first()
		if record:
			session.delete(record)
			session.commit()
			logger.info(f"Text {contents} deleted successfully.")
			return {"message": "Text deleted successfully."}
		else:
			raise HTTPException(status_code=404, detail="contents not found in database.")
	except Exception as e:
		session.rollback()
		logger.error('db error:{}'.format(e))
		raise HTTPException(status_code=500, detail=str(e))

@api.delete("/api/deleteimg", dependencies=[Depends(get_api_key)])
async def delete_img(request: DeleteImagesRequest = Body(...)):
	try:
		for md5 in request.md5s:
			# Query the record with the given md5
			record = session.query(embimg).filter(embimg.md5 == md5).first()
			
			# If the record is found, delete it
			if record:
				session.delete(record)
		
		session.commit()
		logger.info(f"Images deleted successfully.")
		return {"message": "Images deleted successfully."}

	except Exception as e:
		session.rollback()
		logger.error(f"Error deleting images: {str(e)}")
		raise HTTPException(status_code=500, detail="An error occurred while deleting the images.")
	
@api.delete("/api/deleteallimg", dependencies=[Depends(get_api_key)])
async def delete_all_img():
	try:
		session.query(embimg).delete()
		session.commit()
		logger.info(f"All images deleted successfully.")
		return {"message": "All images deleted successfully."}
	except Exception as e:
		session.rollback()
		logger.error(f"Error deleting images: {str(e)}")
		raise HTTPException(status_code=500, detail="An error occurred while deleting the images.")

@api.get("/api/status", dependencies=[Depends(get_api_key)])
async def get_status():
	if model is None:
		text_status, vision_status= False
	else:
		text_status, vision_status= True
	return {'text_status': text_status, 'vision_status': vision_status, 'ram_usage': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}

def main():
	load_model()
	import uvicorn
	uvicorn.run(api, host="0.0.0.0", port=8000)
	
if __name__ == "__main__":
	main()





