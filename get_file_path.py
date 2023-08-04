import os

def get_file_path(root_path, file_list, dir_list, suffixes=['jpeg', 'jpg', 'png']):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list, suffixes)
        else:
            if any(dir_file.endswith(suffix) for suffix in suffixes):  # 检查文件后缀是否在我们的列表中
                file_list.append(dir_file_path)

if __name__ == '__main__':
	root_path = r'./.assets'
	file_list = []
	dir_list = []
	get_file_path(root_path, file_list, dir_list)
	print(file_list)
	print(dir_list)
