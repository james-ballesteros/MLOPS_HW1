FROM --platform=linux/amd64 python:3.10-slim-bullseye

RUN useradd -m appuser
WORKDIR /home/appuser

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m venv /home/appuser/venv
COPY requirements.txt /home/appuser/requirements.txt
RUN /home/appuser/venv/bin/pip install --no-cache-dir -r /home/appuser/requirements.txt

RUN ls -l /home/appuser/venv/bin/uvicorn