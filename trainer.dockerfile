FROM python:3.10.8-slim

RUN mkdir /app
COPY trainer.py /app
COPY neat_config.txt /app
COPY config.json /app
COPY SuperMarioLand.gb /app
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pyboy
RUN python3 -m pip install neat-python
RUN python3 -m pip install numpy
RUN python3 -m pip install Pillow

EXPOSE 37259
ENTRYPOINT [ "python3", "trainer.py" ]
