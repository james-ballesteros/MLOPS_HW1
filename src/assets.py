import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dagster import AssetOut, IOManager, asset, io_manager, multi_asset

# Define a group name for organizational purposes
asset_group_name = "ml_project_assets"

# IO Manager to handle saving and loading of assets
class LocalCSVIOManager(IOManager):
    def handle_output(self, context, obj):
        output_path = f"{context.asset_key.path[-1]}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(output_path, index=False)
        elif isinstance(obj, dict):
            pd.DataFrame([obj]).to_csv(output_path, index=False)

    def load_input(self, context):
        return pd.read_csv(f"{context.asset_key.path[-1]}.csv")

@io_manager
def local_csv_io_manager():
    return LocalCSVIOManager()

# Define assets
@asset(group_name=asset_group_name, compute_kind="pandas", io_manager_key="local_csv_io_manager")
def load_airquality_data():
    """
    Load the AirQualityUCI dataset and return it as a DataFrame.
    """
    source_folder = "data/03_Gold/01_Dev"
    file_path = os.path.join(source_folder, "AirQualityUCI.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    return pd.read_csv(file_path, sep=";", decimal=",")

@multi_asset(
    group_name=asset_group_name,
    compute_kind="scikit-learn",
    outs={"training_data": AssetOut(), "test_data": AssetOut()},
)
def split_data(load_airquality_data):
    """
    Split the AirQualityUCI data into training and test datasets.
    """
    df = load_airquality_data.dropna()
    X = df.drop(columns=["CO(GT)"])  # Replace with actual target column
    y = df["CO(GT)"]  # Replace with actual target column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test)

@asset(group_name=asset_group_name, compute_kind="scikit-learn", io_manager_key="local_csv_io_manager")
def train_model(training_data):
    """
    Train a RandomForestClassifier on the training dataset.
    """
    X_train, y_train = training_data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

@asset(group_name=asset_group_name, compute_kind="scikit-learn")
def evaluate_model(train_model, test_data):
    """
    Evaluate the trained model on the test dataset and return accuracy.
    """
    X_test, y_test = test_data
    predictions = train_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return {"accuracy": accuracy}
