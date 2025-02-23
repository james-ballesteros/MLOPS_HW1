#!/bin/sh

set -e  # Stop on error

# MinIO Configuration
MC_ALIAS="myminio"
MINIO_HOST="http://mlflow_minio:9000"
BUCKET_NAME="mlflow"
ACCESS_KEY="${MINIO_ROOT_USER}"
SECRET_KEY="${MINIO_ROOT_PASSWORD}"

# Function to check MinIO readiness
wait_for_minio() {
  echo "⏳ Waiting for MinIO to be ready..."
  while ! curl -s --fail $MINIO_HOST/minio/health/live | grep -q "ok"; do
    echo "🚨 MinIO not ready, retrying in 5 seconds..."
    sleep 5
  done
  echo "✅ MinIO is ready!"
}

# Install MinIO Client (`mc`)
install_mc() {
  echo "📥 Installing MinIO Client..."
  curl -sSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc
  chmod +x /usr/local/bin/mc
}

# Ensure MinIO is running before continuing
wait_for_minio

# Install MinIO Client
install_mc

# Set MinIO alias
echo "🔧 Configuring MinIO Client..."
mc alias set $MC_ALIAS $MINIO_HOST $ACCESS_KEY $SECRET_KEY

# Ensure bucket exists
echo "📂 Checking if bucket '$BUCKET_NAME' exists..."
if mc ls $MC_ALIAS | awk '{print $NF}' | grep -Fxq "$BUCKET_NAME"; then
  echo "✅ Bucket $BUCKET_NAME already exists."
else
  mc mb $MC_ALIAS/$BUCKET_NAME
  echo "✅ Bucket $BUCKET_NAME created successfully."
fi

echo "🚀 MinIO setup completed!"
