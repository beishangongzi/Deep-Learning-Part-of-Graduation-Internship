FROM tensorflow/tensorflow:latest-gpu

WORKDIR /deep-learning/

COPY requirements.txt  /deep-learning/

RUN pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
# RUN apt-get install graphviz