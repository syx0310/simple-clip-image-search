FROM cnstark/pytorch:2.0.1-py3.9.17-ubuntu20.04
USER root

WORKDIR /app
COPY ./main.py .
COPY ./requirements.txt .

RUN pip3 install torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8000

CMD ["python3", "/app/main.py"]
