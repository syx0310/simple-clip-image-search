import hashlib

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_all_md5(file_paths):
    return [calculate_md5(file_path) for file_path in file_paths]

# text_list=["A dog.", "A car", "A bird"]
if __name__ == '__main__':
	image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
	# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]
	print(get_all_md5(image_paths))
	print(image_paths)
