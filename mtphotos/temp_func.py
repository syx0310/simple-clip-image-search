import json
import requests
from PIL import Image
from io import BytesIO
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, ARRAY, ForeignKey, UniqueConstraint, PrimaryKeyConstraint,VARCHAR, insert, select,desc
import sqlalchemy
from sqlalchemy.sql import text
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer
import yaml
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from loguru import logger

Base = declarative_base()
class emb(Base):
	__tablename__ = 'emb'

	id = mapped_column(VARCHAR, primary_key=True)
	md5 = mapped_column(VARCHAR, primary_key=True)
	filetype = mapped_column(VARCHAR, nullable=True)
	embedding = mapped_column(Vector(768), nullable=True)

	# 设置主键约束
	__table_args__ = (
		PrimaryKeyConstraint('id', 'md5'),
	)

def login(conf):
	url = urljoin(conf['mtphotos']['weburl'], 'auth/login')

	data = {
		"username": conf['mtphotos']['username'],
		"password": conf['mtphotos']['password']
	}
	admin_data = {
		"username": conf['mtphotos']['admin_username'],
		"password": conf['mtphotos']['admin_password']
	}
	headers = {
		"content-type": "application/json",
		"accept": "*/*"
	}
	response = requests.post(url, data=json.dumps(data), headers=headers)
	
	if response.status_code == 201:
		data = response.json()
		conf['mtphotos']['access_token'] = data['access_token']
		conf['mtphotos']['auth_code'] = data['auth_code']
		logger.info(f"User login success with access_token {data['access_token']}")
	else:
		logger.error(f"User login failed with status code {response.status_code}")

	admin_response = requests.post(url, data=json.dumps(admin_data), headers=headers)
	if admin_response.status_code == 201:
		admin_data = admin_response.json()
		conf['mtphotos']['admin_access_token'] = admin_data['access_token']
		conf['mtphotos']['admin_auth_code'] = admin_data['auth_code']
		logger.info(f"Admin login success with access_token {admin_data['access_token']}")
	else:
		logger.error(f"Admin login failed with status code {admin_response.status_code}")

	return conf

def getAllRootFolderId(conf, session, mode='id'):
	url = urljoin(conf['mtphotos']['weburl'], 'folder')

	headers = {
		"Authorization": "Bearer "+ conf['mtphotos']['admin_access_token']
	}
	data = requests.get(url, headers=headers)
	# print(data)
	if data.status_code == 200:
		data = data.json()
		depth_two_dirs = []
		depth_two_ids = []
		for item in data:
			path = item.get('path', '')
			depth = path.count('/')  # Determine depth by counting slashes
			if depth == 2:  # We found a directory at depth 2
				depth_two_dirs.append({'id': str(item['id']), 'path': path})
				depth_two_ids.append(str(item['id']))
		# print(depth_two_dirs)
		if mode == 'dir':
			return depth_two_dirs
		return depth_two_ids
	else:
		logger.error(f"getAllRootFolderId failed with status code {data.status_code}")
		return None
	
def getAllMd5(conf, session, folders=None):
	#"[{'id': '487', 'path': '/photos-1/dev'}]"
	ids = [i['id'] for i in eval(folders)]
	if ids is None:
		return None
	weburl = urljoin(conf['mtphotos']['weburl'], 'gateway/folderFiles/')
	weburls = [urljoin(weburl, i) for i in ids]

	# id = ['487']
	# weburl = urljoin(conf['mtphotos']['weburl'], 'gateway/folderFiles/')
	# weburls = [urljoin(weburl, i) for i in id]
	
	logger.info(f"Will get all md5_id pair from {len(weburls)} urls")

	headers = {
		"Authorization": "Bearer "+ conf['mtphotos']['access_token']
	}

	all_ids_md5 = []
	failed_urls = []
	for url in weburls:
		response = requests.get(url, headers=headers)
		if response.status_code == 200:
			data = response.json()
			ids_md5 = [(str(item['id']), item['MD5'], item['fileType']) for result in data['result'] for item in result['list']]
			ids_md5s = [{'id': id_, 'md5': md5, 'fileType':ft, 'embedding': None} for id_, md5, ft in ids_md5]
			all_ids_md5.extend(ids_md5s)
		else:
			failed_urls.append(url)
			logger.error(f"Request {url} with status code {response.status_code}")
	logger.info(f"Get all md5_id pair from {len(weburls)} urls, failed urls {len(failed_urls)}")
	return all_ids_md5

def insertAllmd5(conf, session):
	all_ids_md5 = getAllMd5(conf, session, conf['mtphotos']['emb_folders'])
	if all_ids_md5 is not None:
		try:
			for row in all_ids_md5:
				sql = text("""
					INSERT INTO emb (id, md5, fileType, embedding)
					VALUES (:id, :md5, :fileType, :embedding)
					ON CONFLICT (id, md5) DO NOTHING
				""")
				
				session.execute(sql, row)

			session.commit()
		except Exception as e:
			logger.error(f"Insert failed with error {e}")
			session.rollback()
		logger.info(f"Insert {len(all_ids_md5)} rows into emb table")
	else:
		logger.error("Insert failed, get all md5 failed")

def fetchBatchImage(urls, cookie, bs):
	assert len(urls) == bs
	images = []
	for url in urls:
		images.append(fetchImage(url, cookie))
	return images

def fetchImage(url, cookie):
	response = requests.get(url, cookies=cookie)
	try:
		if response.status_code == 200:
			image = Image.open(BytesIO(response.content))
			return image
		else:
			# print(f"Request failed with status code {response.status_code}")
			logger.error(f"Request {url} failed with status code {response.status_code}")
			return None
	except Exception as e:
		logger.error(f"Request {url} failed with error {e}")
		return None

def embImages(conf, session, model, processor, updateAll=False, id_md5s=None):
	# if not updateAll, get all ids and md5s that have no embedding and get embedding
	# id_md5s = [{'id': id, 'md5': md5, 'fileType': fileType}]
	if updateAll:
		id_md5s = getAllMd5(conf, session, conf['mtphotos']['emb_folders'])
		# 去除embedding列
		try:
			id_md5s = [{'id': i['id'], 'md5': i['md5'], 'fileType': i['fileType']} for i in id_md5s]
		except Exception as e:
			logger.error(f"Get all md5 failed with error {e}")
			return 1
	if id_md5s is None:
		try:
			id_md5s = getEmbIds(conf, session)
		except Exception as e:
			logger.error(f"Get embedding ids failed with error {e}")
			return 2
	
	# remove mp4
	id_md5s = [i for i in id_md5s if i['fileType'] != 'MP4' ]

	urls = [urljoin(conf['mtphotos']['weburl'], 'gateway/file/') + i['id'] + "/" + i['md5'] for i in id_md5s]
	# get image
	bc = len(urls)//conf['model']['batch_size']
	lastbs = len(urls)%conf['model']['batch_size']
	cookies = {
		"auth_code": conf['mtphotos']['auth_code']
	}
	bs = conf['model']['batch_size']
	error_index = []
	len_updated = 0
	for i in range(bc):
		if i == bc-1:
			images = fetchBatchImage(urls[i*bs:(i+1)*bs+lastbs], cookies, bs+lastbs)
		else:
			images = fetchBatchImage(urls[i*bs:(i+1)*bs], cookies, bs)

		# 删除空值
		avali_index = [j + i*bs for j, im in enumerate(images) if im is not None]
		len_updated += len(avali_index)
		error_index.extend([j + i*bs for j, im in enumerate(images) if im is None])
		images = [im for im in images if im is not None]
		# get embedding and update embedding in db
		with torch.no_grad():
			inputs = processor(images=images, return_tensors="pt", padding=True).to(conf['model']['device'])
			image_features = model.get_image_features(**inputs)
			print(image_features[0].shape)
			

			for k,j in enumerate(avali_index):
				sql = text("""
					UPDATE emb
					SET embedding = :embedding
					WHERE id = :id AND md5 = :md5
				""")
				session.execute(sql, {'id': id_md5s[j]['id'], 'md5': id_md5s[j]['md5'], 'embedding': image_features[k].tolist()})
			session.commit()
	logger.info(f"Update {len_updated} rows in emb table")
	if len(error_index) > 0:
		logger.error(f"Error count {len(error_index)}")
	return 0

def getEmbIds(conf, session):
	# get ids and md5s that have no embedding
	query = (
		select(emb.id, emb.md5, emb.filetype)
		.where(emb.embedding == None)
	)
	result = session.execute(query).all()
	result = [{'id': i[0], 'md5': i[1], 'fileType': i[2]} for i in result]
	return result

def searchDB(conf, session, model, processor, tokenizer, cond):
	with torch.no_grad():
		if isinstance(cond, str):
			input = tokenizer(cond, return_tensors="pt", padding=True).to(conf['model']['device'])
			embeddings = model.get_text_features(**input)
		elif isinstance(cond, Image.Image):
			input = processor(images=cond, return_tensors="pt")
			embeddings = model.get_image_features(**input.to(conf['model']['device']))

	query = (
		select(emb.id, emb.md5)
		.where(emb.embedding.isnot(None))
		.order_by(emb.embedding.cosine_distance(embeddings[0]))
	)
	result = session.execute(query).all()
	out_dis = session.scalars(
		select(emb.embedding.cosine_distance(embeddings[0]))
		.where(emb.embedding.isnot(None))
		.order_by(desc(emb.embedding.cosine_distance(embeddings[0])))).all()
	# print(result)
	logger.info(f"Search result {result}")
	logger.info(f"Search result {out_dis}")
	return result, out_dis

def syncDB(conf, session):
	# 1. delete data that not exist/not match
	# 2. insert new data
	all_md5s = getAllMd5(conf, session, conf['mtphotos']['emb_folders'])
	
	# select all data expect embedding
	query = (
		select(emb.id, emb.md5, emb.filetype)
	)

	dbdata = session.execute(query).all()
	dbdata = [{'id': i[0], 'md5': i[1], 'fileType': i[2], 'embedding': None} for i in dbdata]
	set1 = set(frozenset(i.items()) for i in all_md5s)
	set2 = set(frozenset(i.items()) for i in dbdata)

	newdata = [i for i in all_md5s if frozenset(i.items()) not in set2]
	deldata = [i for i in dbdata if frozenset(i.items()) not in set1]

	logger.info(f"New data {len(newdata)}, {newdata}")
	logger.info(f"Del data {len(deldata)}, {deldata}")

	# delete data
	if len(deldata) > 0:
		try:
			for row in deldata:
				sql = text("""
					DELETE FROM emb
					WHERE id = :id AND md5 = :md5
				""")
				
				session.execute(sql, row)

			session.commit()
		except Exception as e:
			logger.error(f"Insert failed with error {e}")
			session.rollback()
	# insert data
	if len(newdata) > 0:
		try:
			for row in newdata:
				sql = text("""
					INSERT INTO emb (id, md5, fileType, embedding)
					VALUES (:id, :md5, :fileType, :embedding)
					ON CONFLICT (id, md5) DO NOTHING
				""")
				
				session.execute(sql, row)

			session.commit()
		except Exception as e:
			logger.error(f"Insert failed with error {e}")
			session.rollback()

if __name__ == '__main__':
	logger.add("dev.log", rotation="100 MB") 
	conf = yaml.load(open('config.yml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
	logger.info('config loaded{}'.format(conf))
	login(conf)
	# print('conf', conf)
	# exit()
	DATABASE_URL = conf['devdb']['dburl']
	engine = create_engine(DATABASE_URL, echo=True)

	session = sessionmaker(bind=engine)()

	syncDB(conf, session)
	# insertAllmd5(conf, session)

	# getAllRootFolderId(conf, session)

	# getEmbIds(conf,session)

	# model = ChineseCLIPModel.from_pretrained(conf['model']['model_name']).to(conf['model']['device'])
	# processor = ChineseCLIPProcessor.from_pretrained(conf['model']['model_name'])
	# tokenizer = AutoTokenizer.from_pretrained(conf['model']['model_name'])

	# result, dis = searchDB(conf, session, model, processor, tokenizer, '两个人')
	# urls = [urljoin(conf['mtphotos']['weburl'], 'gateway/file/') + i[0] + "/" + i[1] for i in result]
	# print(urls)
	exit()
	try:
		insertAllmd5(conf, session)
	except Exception as e:
		print(e)
		session.rollback()


