FROM python:3.7.6-buster

COPY . /app

WORKDIR /app

ARG AZURE_CONN_STR

ARG MONGODB_URL

ARG MLFLOW_TRACKING_URI

ARG MLFLOW_TRACKING_USERNAME

ARG MLFLOW_TRACKING_PASSWORD

ENV AZURE_CONN_STR $AZURE_CONN_STR

ENV MONGODB_URL ${MONGODB_URL}

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}}

ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}

ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

RUN pip install -r requirements.txt

CMD ["python","main.py"]