FROM python:3.7-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

WORKDIR $APP_HOME

# needs to be compiled for latest cuda to work on high end GPUs
RUN pip3 install --no-cache-dir torch
RUN pip3 install --no-cache-dir kfserving
RUN pip3 install --no-cache-dir transformers
RUN pip3 install --no-cache-dir accelerate

COPY . /app

CMD ["python3", "main.py"]