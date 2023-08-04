import gradio as gr
from querydb import query_database
import torch
from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, CLIPModel, CLIPProcessor
import time

def main(lang='zh'):
	load_start = time.time()
	def getresult(text):
		start = time.time()
		results = query_database(text, model, tokenizer, processor, device)
		outtime = time.time() - start

		file_paths, distances = results

		return *file_paths, distances, outtime

	def main_tab():
		with gr.Column():
			prompt = gr.Textbox(label="text")
			button = gr.Button(text="Search")
			
			with gr.Row():
				img1 = gr.Image(type='filepath',label="Image 1")
				img2 = gr.Image(type='filepath',label="Image 2")
				img3 = gr.Image(type='filepath',label="Image 3")
				img4 = gr.Image(type='filepath',label="Image 4")
				img5 = gr.Image(type='filepath',label="Image 5")
			distances = gr.Textbox(label="Cosine Distances")
			time = gr.Textbox(label="Time taken")
			button.click(getresult, inputs=prompt, outputs=[img1,img2,img3,img4,img5,distances,time])

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if lang == 'zh':
		model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px").to(device)
		processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
		tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-large-patch14-336px")
	elif lang == 'en':
		model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
		processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
		tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
	load_end = time.time()
	print(f"Loaded model in {load_end - load_start} seconds")
	with gr.Blocks() as ui:
		with gr.Tab("Results"):
			main_tab()
			
	ui.launch(server_name='0.0.0.0')
	
if __name__ == '__main__':
	main()