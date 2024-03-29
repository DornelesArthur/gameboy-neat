FROM python:3.10.8-slim

RUN mkdir /app
COPY manager.py /app

WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install neat-python

COPY neat_config.txt /app
COPY config.json /app

EXPOSE 37259
ENTRYPOINT [ "python3", "manager.py" ]