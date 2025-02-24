# MLflow Docker Setup with PostgreSQL and MinIO

## **Prerequisites**
Before proceeding, ensure you have:

- [Docker](https://www.docker.com/get-started) installed (v20+ recommended)
- [Docker Compose](https://docs.docker.com/compose/) installed (v2+ recommended)

---

## **1. Setup the Project Structure**
Ensure the following folder structure exists:

```plaintext
.
├── docker-compose.yaml
├── mlflow
│   ├── Dockerfile  # Place the MLflow Dockerfile inside mlflow folder
├── .env.example
```

Create the required directories:

```bash
mkdir -p postgres_data minio_data mlflow
```

Move the `Dockerfile` inside the `mlflow` folder:

```bash
mv Dockerfile mlflow/
```

---

## **2. Copy and Configure Environment Variables**
Rename `.env.example` to `.env`:

```bash
cp .env.example .env
```

Now, update the `.env` file with correct values.

---

## **3. Start Services for the First Time**
Run:

```bash
docker-compose up -d --build
```

---

## **4. Generate MinIO Access and Secret Keys**
1. Open the MinIO console in your browser:  
   **[http://localhost:9001](http://localhost:9001)**  
2. Log in with:
   - **Username:** `minio_user`
   - **Password:** `minio_pwd`
3. Create a **bucket** named `mlflow`
4. Generate new **Access Key** and **Secret Key** for MLflow
5. Copy and update the `.env` file with the generated **MinIO Access Key** and **Secret Key**
6. Restart the services:

```bash
docker-compose down -v
docker-compose up -d --build
```

---

## **5. Verify the Setup**
### **Check Running Containers**
```bash
docker ps
```
You should see three running containers:
- `mlflow_postgres`
- `mlflow_minio`
- `mlflow_server`

### **Verify PostgreSQL**
```bash
docker logs mlflow_postgres
```
Ensure no errors appear.

### **Verify MinIO Health**
```bash
curl -f http://localhost:9000/minio/health/live && echo "MinIO is healthy"
```

### **Verify MLflow**
```bash
curl -f http://localhost:5050/ && echo "MLflow is running"
```

---

## **6. Testing MLflow Tracking**
Run the following Python script to test MLflow:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run():
    mlflow.log_param("param1", 42)
    mlflow.log_metric("accuracy", 0.95)

print("Test run logged successfully!")
```

You should see the logged run in the MLflow UI:  
[http://localhost:5000](http://localhost:5000)

---

## **7. Stop and Remove Containers**
To clean up:

```bash
docker-compose down -v
```

---

## **Troubleshooting**

### MLFlow
- MLFlow fails to start
    * Ensure `MLFLOW_S3_ENDPOINT_URL` is correctly set in `.env`

### MinIO
- MinIO bucket not found
    * Make sure you created the `mlflow` bucket inside the MinIO web console
- MinIO failing to start due to missing port
    * Error Message: `Unable to start listening on console port: address 0.0.0.0: missing port in address`
    * Set `MINIO_CONSOLE_ADDRESS` to `0.0.0.0:9001` and `MINIO_ADDRESS` to `0.0.0.0:9000`

### PostgreSQL
- PostGres connection failing
    * Confirm `PG_USER`, `PG_PASSWORD`, and `PG_DATABASE` in `.env` match those in `docker-compose.yaml`

---

## **Final Notes**
- This setup is intended for **local development**.
- For **production**, consider enabling **MinIO TLS** and securing credentials.

---