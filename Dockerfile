FROM python:3.9.7

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 8080 8888
CMD [ "python", "server.py"]
