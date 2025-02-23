import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Setup MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "AirQuality_Model_Training"
mlflow.set_experiment(experiment_name)

# Load Data
df = pd.read_csv("/home/appuser/feature_engineering.csv", sep=",", decimal=".")

# Features & Target assignment
X = df.select_dtypes(include=["number"]).drop(columns=["CO(GT)"])  # Drop target column
y = df["CO(GT)"]  # Target variable

# Ensure the dataset contains only numeric values before training
assert X.select_dtypes(exclude=["number"]).empty, "Non-numeric columns detected in X!"

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different model configurations
best_r2 = -1  # Track best R2 score
best_run_id = None

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

# **Register the best model**
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "best_air_quality_model")
    print("Best model registered successfully!")
