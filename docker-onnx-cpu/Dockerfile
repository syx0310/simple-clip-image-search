FROM python:3.9.18-slim-bullseye

WORKDIR /app

COPY main.py ./
COPY utils/ ./utils/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8000

CMD ["python", "main.py"]
