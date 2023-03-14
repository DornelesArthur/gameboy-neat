FROM python:3.10.8-slim

RUN mkdir /app
COPY manager.py /app
COPY neat_config.txt /
COPY config.json /

WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install neat-python

EXPOSE 37259
ENTRYPOINT [ "python3", "manager.py" ]