import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

# 1️⃣ **Set MLflow Tracking URI** (Ensure this matches your MLflow server)
mlflow.set_tracking_uri("http://mlflow_server:5050")

# 2️⃣ **Set experiment name**
experiment_name = "AirQuality_Model_Training"

# Check if experiment exists, create if not
client = MlflowClient()
existing_experiments = [exp.name for exp in client.search_experiments()]
if experiment_name not in existing_experiments:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# 3️⃣ **Load Data**
df = pd.read_csv("/home/appuser/feature_engineering.csv", sep=",", decimal=".")

# 4️⃣ **Features & Target assignment**
X = df.select_dtypes(include=["number"]).drop(columns=["CO(GT)"])  # Drop target column
y = df["CO(GT)"]  # Target variable

# Ensure the dataset contains only numeric values before training
assert X.select_dtypes(exclude=["number"]).empty, "Non-numeric columns detected in X!"

# 5️⃣ **Train/Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ **Train multiple models and log the best one**
best_r2 = -1
best_run_id = None
model_name = "best_air_quality_model"

for n_estimators in [50, 100, 150]:  # Try different hyperparameters
    with mlflow.start_run():
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained with n_estimators={n_estimators}, R2 Score: {r2:.4f}")

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_run_id = mlflow.active_run().info.run_id

print(f"Best Model Run ID: {best_run_id} with R2 Score: {best_r2}")

# 7️⃣ **Register the best model**
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, model_name)
    print("Best model registered successfully!")

    # 8️⃣ **Move best model to "Production" stage**
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    print(f"Model version {latest_version} transitioned to Production.")
