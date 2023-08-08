import gradio as gr
from querydb import query_database
from imagebind import data
import torch
from PIL import Image
import requests
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, CLIPModel, CLIPProcessor
import time
import yaml
from temp_func import *
from loguru import logger
import requests
import shutil
from pykwalify.core import Core

	
def main():
	def search(cond, limit):
		start = time.time()
		result, dis = searchDB(conf, session, model, processor, tokenizer, cond)
		outtime = time.time() - start
		urls = [urljoin(conf['mtphotos']['weburl'], 'gateway/file/') + i[0] + "/" + i[1] for i in result]
		return urls[0:limit], dis[0:limit], outtime
	def reloadConfig(conf):
		with open('config.yaml', 'r') as f:
			conf = yaml.load(f, Loader=yaml.FullLoader)
		return conf
	def login1(text):
		return login(conf)
	def getAllRootFolderId1(_):
		return getAllRootFolderId(conf, session, 'dir')
	def embImages1(_):
		embImages(conf, session, model, processor)
		return "Done"
	def embImagesAll(_):
		embImages(conf, session, model, processor, True)
		return "Done"
	def writeConfig(_):
		# use yaml to write config
		schema = """
type: map
mapping:
  db:
    type: map
    mapping:
      dburl: {type: str, required: true}
  devdb:
    type: map
    mapping:
      dburl: {type: str, required: true}
  model:
    type: map
    mapping:
      batch_size: {type: int, required: true}
      device: {type: str, required: true}
      model_name: {type: str, required: true}
  mtphotos:
    type: map
    mapping:
      access_token: {type: str, required: true}
      admin_access_token: {type: str, required: true}
      admin_auth_code: {type: str, required: true}
      admin_password: {type: str, required: true}
      admin_username: {type: str, required: true}
      auth_code: {type: str, required: true}
      emb_folders: {type: str, required: true}
      password: {type: str, required: true}
      username: {type: str, required: true}
      weburl: {type: str, required: true}
"""
		shutil.copy2("./config.yml", "./config.yml.bak")
		with open("./config.yml", "w") as f:
			try:
				c = Core(source_data=yaml.load(_, Loader=yaml.FullLoader), schema_data=yaml.load(schema, Loader=yaml.FullLoader))
				try:
					c.validate(raise_exception=True)
				except Exception as e:
					logger.error("Config not written {}".format(e))
					restoreConfig(None)
					return "Config not written {}".format(e)
				yaml.dump(conf, f)
				logger.info("Config written")
			except Exception as e:
				logger.error("Config not written {}".format(e))
				restoreConfig(None)
				return "Config not written {}".format(e)
		reloadConfig(conf)
		return "Config written"
	
	def restoreConfig(_):
		shutil.copy2("./config.yml.bak", "./config.yml")
		reloadConfig(conf)
		logger.info("Config restored")
	def showConfig(_):
		return conf


	def sync(_):
		try:
			syncDB(conf, session)
			return "Done"
		except Exception as e:
			return "Error {}".format(e)

	def emb_tab():
		with gr.Column():
			text = gr.Textbox(label="Output", lines = 5)
			text2 = gr.Textbox(label="Input", lines = 5)
			with gr.Row():
				buttonLogin = gr.Button("Login")
				buttonLogin.click(login1, inputs=text, outputs=[text])
				buttonListRootFolders = gr.Button("List root folders")
				buttonListRootFolders.click(getAllRootFolderId1, inputs=text, outputs=[text])
				
				buttonEmbEmpty = gr.Button("Emb Empty")
				buttonEmbEmpty.click(embImages1, inputs=text, outputs=[text])
				buttonEmbAll = gr.Button("Emb All")
				buttonEmbAll.click(embImagesAll, inputs=text, outputs=[text])

			with gr.Row():
				buttonWriteConfig = gr.Button("Write Config")
				buttonWriteConfig.click(writeConfig, inputs=text2, outputs=[text])
				buttonRestoreConfig = gr.Button("Restore Config")
				buttonRestoreConfig.click(restoreConfig, inputs=text, outputs=[text])
				buttonShowConfig = gr.Button("Show Config")
				buttonShowConfig.click(showConfig, inputs=text, outputs=[text])
				buttonSync = gr.Button("Sync")
				buttonSync.click(sync, inputs=text, outputs=[text])
			# with gr.Row():
			# 	# buttonSelectAllFolders = gr.Button("Select All Folders")
			# 	# buttonSelectAllFolders.click(selectAllFolders, inputs=text, outputs=[text])

	def main_tab():
		with gr.Column():
			cond = gr.Textbox(label="text")
			# conf = gr.Textbox(label="Config")
			with gr.Row():
				result = gr.Textbox(label="Result")
				with gr.Column():
					selectLen = gr.Radio([5,10,15,20], label="Select")
					searchButton = gr.Button("Search")
			distances = gr.Textbox(label="Cosine Distances")
			outtime = gr.Textbox(label="Time taken")
			
			# time = gr.Textbox(label="Time taken")
			searchButton.click(search, inputs=[cond, selectLen], outputs=[result, distances, outtime])

	st = time.time()
	logger.add("dev.log", rotation="100 MB") 
	conf = yaml.load(open('config.yml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
	logger.info('config loaded{}'.format(conf))

	login(conf)

	DATABASE_URL = conf['devdb']['dburl']
	engine = create_engine(DATABASE_URL, echo=True)

	Base = declarative_base()

	class emb(Base):
		__tablename__ = 'emb'

		id = mapped_column(VARCHAR, primary_key=True)
		md5 = mapped_column(VARCHAR, primary_key=True)
		filetype = mapped_column(VARCHAR, nullable=True)
		embedding = mapped_column(Vector(768), nullable=True)

		__table_args__ = (
			PrimaryKeyConstraint('id', 'md5'),
		)
	
	session = sessionmaker(bind=engine)()
	# Test db
	try:
		session.query(emb)
		logger.info('db connected')
	except Exception as e:
		logger.error('db error:{}'.format(e))
		exit()

	if 'cuda' in conf['model']['device']:
		if torch.cuda.is_available():
			logger.info('Using GPU')
		else:
			logger.error('GPU not available, please check your config')
			exit()

	model = ChineseCLIPModel.from_pretrained(conf['model']['model_name']).to(conf['model']['device'])
	processor = ChineseCLIPProcessor.from_pretrained(conf['model']['model_name'])
	tokenizer = AutoTokenizer.from_pretrained(conf['model']['model_name'])

	logger.info('All loaded in {} seconds'.format(time.time() - st))

	with gr.Blocks() as ui:
		with gr.Tab("Search"):
			main_tab()
		with gr.Tab("Settings"):
			emb_tab()

	ui.launch()

if __name__ == '__main__':
	main()