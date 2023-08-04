# A simple image search engine using pgvector and clip

## Files
```
.
├── all_md5.py # 生成所有图片的md5值
├── api.py # 获取图片/文字特征的简单api
├── .assets # 测试用图片文件夹
├── clip-test.py # 测试clip模型运行
├── config.yml # 配置文件
├── docker-compose.yml # pgvector数据库配置文件
├── get_file_path.py # 获取文件夹下所有图片文件的路径
├── init.sql # pgvector数据库初始化
├── process_img.py # 将图片转换为向量并存入向量数据库
├── querydb.py  # 查询向量数据库
├── requirements.txt # python依赖
├── webui.py # 一个简单webui
└── write_md5_path_pair.py # 将图片的md5值和路径写入数据库
```

## run this project
```bash
conda env create -f environment.yaml
conda activate imagebind

```

### If you only want use clip to get image feature or text feature

```bash
uvicorn api:app --reload
```

Get text feature
```bash
curl --location --request POST '/embtext' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
--form 'text="<text>"'
```

Get image feature
```bash
curl --location --request POST '/embimage' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
--form 'file=@"<file>"'
```

### If you want to try webui

```bash
mkdir data

docker-compose up -d
```

edit config.yml to fit your environment  

connect to pgvector database and run init.sql  

```bash
python write_md5_path_pair.py # write md5 and path to database
python process_img.py # process image and write vector to database
python webui.py # run webui
```

