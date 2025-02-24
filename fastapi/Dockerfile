# FastAPI Dockerfile
FROM python:3.9

WORKDIR /code

# Copy dependencies file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files from the fastapi folder
COPY . /code

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:8000"]
