FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt
COPY ./app /code/app
COPY ./backend /code/backend

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
