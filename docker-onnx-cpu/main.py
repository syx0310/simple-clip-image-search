from PIL import Image
from io import BytesIO
from sqlalchemy import create_engine, Column, Integer, String, PrimaryKeyConstraint,VARCHAR, insert, select,desc,text

from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import onnxruntime
from loguru import logger
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Body
from typing import List
from pydantic import BaseModel
import os
from time import sleep
import psutil
import sys
from utils.utils import image_processor, tokenize_numpy
import argparse
from threading import Timer

api = FastAPI()

def load_conf_from_env():
	# Load config from environment variables
	conf = {}
	conf['dburl'] = os.environ.get('DBURL', 'postgresql://mtphotos:mtphotos@clip-db:5432/mtphotos')
	conf['model'] = os.environ.get('MODEL', 'ViT-B-16')
	conf['model_path'] = os.environ.get('MODEL_PATH', '/data/model')
	conf['model_img_path'] = os.path.join(conf['model_path'], conf['model'])+'.img.fp32.onnx'
	conf['model_txt_path'] = os.path.join(conf['model_path'], conf['model'])+'.txt.fp32.onnx'
	conf['device'] = os.environ.get('DEVICE', 'cpu')
	conf['clip_api_key'] = os.environ.get('CLIP_API_KEY', 'YOUR_SECRET_API_KEY')
	conf['log_level'] = os.environ.get('LOG_LEVEL', 'INFO')
	return conf

conf = load_conf_from_env()
logger.add("/data/main.log", rotation="100 MB", level=conf['log_level'])

if conf['model'] == 'ViT-B-16':
	EMB_DIM = 512
else:
	EMB_DIM = 768

if '336' in conf['model']:
	IMG_SIZE = 336
else:
	IMG_SIZE = 224

processor = image_processor
tokenizer = tokenize_numpy

Base = declarative_base()

class embimg(Base):
	__tablename__ = 'embimg'

	md5 = mapped_column(VARCHAR, primary_key=True)
	embedding = mapped_column(Vector(EMB_DIM), nullable=True)

	__table_args__ = (
		PrimaryKeyConstraint('md5'),
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

vision_model = None
text_model = None

def get_api_key(
    api_key: str = Header(None, alias="api_key"),
):
    if api_key != conf['clip_api_key']:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

def load_vis_model():
	global vision_model
	img_sess_options = onnxruntime.SessionOptions()
	img_run_options = onnxruntime.RunOptions()
	img_run_options.log_severity_level = 2
	vision_model = onnxruntime.InferenceSession(conf['model_img_path'],
											sess_options=img_sess_options,
											providers=["CPUExecutionProvider"])

def load_text_model():
	global text_model
	txt_sess_options = onnxruntime.SessionOptions()
	txt_run_options = onnxruntime.RunOptions()
	txt_run_options.log_severity_level = 2
	text_model = onnxruntime.InferenceSession(conf['model_txt_path'],
											sess_options=txt_sess_options,
											providers=["CPUExecutionProvider"])

def check_model(type):
	if type == 'text':
		if text_model is None:
			return False
	else:
		if vision_model is None:
			return False
	return True

async def write_imgdb(md5, embedding):
	try:
		record = embimg(md5=md5, embedding=embedding)
		session.merge(record)
		session.commit()
	except Exception as e:
		logger.error('db error:{}'.format(e))
		session.rollback()
		raise HTTPException(status_code=500, detail=str(e))

async def searchDB(session, model, processor, tokenizer, cond):

	if isinstance(cond, str):
		if check_model('text')==False:
			logger.error(f"Text model not loaded")
			raise HTTPException(status_code=500, detail="Text model not loaded while searching")
		input = tokenizer([cond], 52)
		embeddings = text_model.run(["unnorm_text_features"], {"text":input})[0]
	elif isinstance(cond, Image.Image):
		if check_model('vision')==False:
			logger.error(f"Vision model not loaded")
			raise HTTPException(status_code=500, detail="Vision model not loaded while searching")
		input = processor([cond], image_size=IMG_SIZE)
		embeddings = vision_model.run(["unnorm_image_features"], {"image":input})[0]
	if embeddings[0].shape[0] != EMB_DIM:
		logger.error(f"Embedding dimension mismatch, please check your model.")
		raise HTTPException(status_code=500, detail="Embedding dimension mismatch, please check your model.")
	query = (
		select(embimg.md5)
		.where(embimg.embedding.isnot(None))
		.order_by(embimg.embedding.cosine_distance(embeddings[0].tolist()))
	)
	result = session.execute(query).all()
	out_dis = session.scalars(
		select(embimg.embedding.cosine_distance(embeddings[0].tolist()))
		.where(embimg.embedding.isnot(None))
		.order_by(desc(embimg.embedding.cosine_distance(embeddings[0].tolist())))).all()
	logger.info(f"Search result {len(result)}")
	# logger.info(f"Search result {len(out_dis)}")
	return result, out_dis


@api.get("/api/initdb", dependencies=[Depends(get_api_key)])
async def initdb():
	init_sqls = [
				'DROP TABLE IF EXISTS embimg;',
				'DROP EXTENSION IF EXISTS vector;',
			  'CREATE EXTENSION vector;',
			  f'CREATE TABLE embimg (md5 VARCHAR ,embedding VECTOR({EMB_DIM}), CONSTRAINT pk_embimg PRIMARY KEY (md5));'
				]

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
	if vision_model is None:
		# return HTTPException(status_code=500, detail="Vision model not loaded")
		return {"status": f"Error: vision model not loaded", 'count': 0 }
	try:
		images = [Image.open(BytesIO(file.file.read())) for file in files]
	except Exception as e:
		logger.error(f"Error: file open related error occurred: {str(e)}")
		return {"status": f"Error: file open related error occurred: {str(e)}", 'count': 0 }
	try: 
		inputs = processor(images, image_size=IMG_SIZE)
		image_features = []
		for i in range(len(inputs)):
			input = inputs[i:i+1,:,:,:]
			image_feature = vision_model.run(["unnorm_image_features"], {"image": input})[0].tolist()[0]
			image_features.append(image_feature)

		for md5, embedding in zip(md5s, image_features):
			try:
				await write_imgdb(md5, embedding)
			except Exception as e:
				logger.error(f"Error: db error occurred: {str(e)}")
				return {"status": f"Error: db error occurred: {str(e)}", 'count': 0 }
		logger.info(f"Image {md5s} processed successfully.")
		logger.debug('ram usage: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
		return {'status': 'done', 'count': len(image_features)}
	except RuntimeError as e:
		if "out of memory" in str(e):
			logger.error(f"Error: GPU memory overflow. Please try with smaller batch size.")
			return {"status": "Error: GPU memory overflow. Please try with smaller batch size.", 'count': 0 }
		else:
			logger.error(f"Error: unexpected error occurred: {str(e)}")
			return {"status": f"Error: unexpected error occurred: {str(e)}", 'count': 0 }

@api.post("/api/search", dependencies=[Depends(get_api_key)])
async def search(request: SearchRequest):
	text = request.text
	start = request.start
	end = request.end
	if end < start and end > 0:
		end = start + 1
	if start < 0:
		start = 0
	result, out_dis = await searchDB(session, text_model, processor, tokenizer, text)
	if end < 0 or end > len(result):
		end = len(result)
	print('ram usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
	md5 = [i[0] for i in result[start:end]]
	return {'md5': md5, 'dis': out_dis[start:end]}

@api.get("/api/cuda", dependencies=[Depends(get_api_key)])
async def get_cuda():
	return {'cuda': 'False'}

@api.get("/api/device", dependencies=[Depends(get_api_key)])
async def get_device():
	model = text_model if text_model is not None else vision_model

	return {'device': str(model._providers), 'totalvram': 0}
	
def restart(model):
	python = sys.executable
	args = sys.argv[:]

    # 检查 --model 参数是否已存在
	if '--model' in args:
		# 如果存在，找到它的索引并更新其值
		index = args.index('--model')
		args[index + 1] = model
	else:
		# 如果不存在，添加 --model 参数和其值
		args.append('--model')
		args.append(model)
	Timer(1, lambda: os.execl(python, python, *args)).start()

@api.post("/api/switchmodel", dependencies=[Depends(get_api_key)])
def switchmodel(request_model:RestartModel):
	global vision_model, text_model
	if request_model.model == 'text':
		restart('text')
	elif request_model.model == 'vision':
		restart('vision')
	else:
		logger.error(f"Error: model {request_model.model} not found")
		raise HTTPException(status_code=500, detail=f"Error: model {request_model.model} not found")

	return {"message": f"Restarting with {request_model.model} model"}


@api.get("/api/move2cpu", dependencies=[Depends(get_api_key)])
def move2cpu():
	return {'status': 'not supported', 'device': 'cpu'}

@api.get("/api/move2gpu", dependencies=[Depends(get_api_key)])
def move2gpu():
	return {'status': 'not supported', 'device': 'cpu'}

@api.get("/api/getallimagemd5", dependencies=[Depends(get_api_key)])
async def getAllImageMD5():
	query = (
		select(embimg.md5)
		.where(embimg.embedding.isnot(None))
	)
	result = session.execute(query).all()
	md5 = [i[0] for i in result]
	return {'md5': md5}

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
	if text_model is None:
		text_status = False
	else:
		text_status = True
	if vision_model is None:
		vision_status = False
	else:
		vision_status = True
	return {'text_status': text_status, 'vision_status': vision_status, 'ram_usage': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}

def main():
	parser = argparse.ArgumentParser(description='CLIP API')
	parser.add_argument('--model', type=str, default='text', help='select model to load')
	args = parser.parse_args()
	if args.model == 'text':
		load_text_model()
	elif args.model == 'vision':
		load_vis_model()
	else:
		logger.error(f"Error: model {args.model} not found")
		exit()
	import uvicorn
	uvicorn.run(api, host="0.0.0.0", port=8000)
	
if __name__ == "__main__":
	main()





