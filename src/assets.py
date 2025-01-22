import pandas as pd
import os
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dagster import AssetOut, IOManager, asset, io_manager, multi_asset


# Define group name
asset_group_name = "ml_project_assets"

# IO Manager
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
    file_path = os.path.join(source_folder, "AirQualityUCI_Final.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    df = pd.read_csv(file_path, sep=";", decimal=",")
    if df.empty:
        raise ValueError("The dataset is empty after loading. Please check the file.")
    return df

@asset(group_name=asset_group_name, compute_kind="pandas", io_manager_key="local_csv_io_manager")
def preprocess_data(load_airquality_data):
    """
    Preprocess the data by handling missing values and scaling numeric features.
    """
    df = load_airquality_data

    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns

    # Drop numeric columns with excessive missing data
    df_numeric = df[numeric_columns].dropna(thresh=len(df) * 0.5, axis=1)

    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy="mean")
    df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Scale numeric features
    scaler = StandardScaler()
    df_numeric = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Retain non-numeric columns
    df_non_numeric = df[non_numeric_columns]

    # Combine numeric and non-numeric columns
    df = pd.concat([df_numeric, df_non_numeric.reset_index(drop=True)], axis=1)

    if df.empty:
        raise ValueError("The dataset is empty after preprocessing. Please check the preprocessing logic.")

    return df

@asset(group_name=asset_group_name, compute_kind="pandas", io_manager_key="local_csv_io_manager")
def feature_engineering(preprocess_data):
    """
    Perform feature engineering on the preprocessed data.
    """
    df = preprocess_data

    # Create a new feature
    if "CO(GT)" in df.columns and "RH" in df.columns:
        df["CO_Norm"] = df["CO(GT)"] / df["RH"]  # Normalize CO by relative humidity

    if df.empty:
        raise ValueError("The dataset is empty after feature engineering. Please check the feature engineering logic.")
    
    return df

@multi_asset(
    group_name=asset_group_name,
    compute_kind="scikit-learn",
    outs={"training_data": AssetOut(), "test_data": AssetOut()},
)
def split_data(feature_engineering):
    """
    Split the dataset into training and testing sets.
    """
    df = feature_engineering

    # Ensure the DataFrame is not empty
    if df.empty:
        raise ValueError("The dataset is empty and cannot be split. Please check the upstream assets.")

    # Ensure the target column exists
    if "CO(GT)" not in df.columns:
        raise ValueError("The target column 'CO(GT)' is missing in the dataset.")

    # Filter numeric columns for features
    numeric_columns = df.select_dtypes(include=["number"]).columns
    X = df[numeric_columns].drop(columns=["CO(GT)"])
    y = df["CO(GT)"]

    if len(X) == 0 or len(y) == 0:
        raise ValueError("The dataset contains no samples to split.")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("The train or test set is empty. Adjust test_size or check the dataset.")

    return (X_train, y_train), (X_test, y_test)

@asset(group_name=asset_group_name, compute_kind="scikit-learn")
def train_model(training_data):
    """
    Train a RandomForestRegressor on the training dataset.

    Args:
        training_data (tuple): A tuple containing (X_train, y_train).

    Returns:
        RandomForestRegressor: A trained RandomForestRegressor model.
    """
    X_train, y_train = training_data

    # Ensure X_train and y_train are not empty
    if X_train.empty or len(y_train) == 0:
        raise ValueError("The training dataset is empty. Please check the upstream assets.")

    # Check for consistent sizes between X_train and y_train
    if len(X_train) != len(y_train):
        raise ValueError("Mismatched lengths: X_train has {} rows, while y_train has {} rows.".format(
            len(X_train), len(y_train)))

    # Train the regression model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model

@asset(group_name=asset_group_name, compute_kind="scikit-learn")
def evaluate_model(train_model, test_data):
    """
    Evaluate the trained regression model on the test dataset.

    Args:
        train_model (RandomForestRegressor): Trained model.
        test_data (tuple): A tuple containing (X_test, y_test).

    Returns:
        dict: Evaluation metrics including MSE, MAE, and R² score.
    """
    X_test, y_test = test_data

    # Ensure X_test and y_test are not empty
    if X_test.empty or len(y_test) == 0:
        raise ValueError("The test dataset is empty. Please check the upstream assets.")

    # Generate predictions
    predictions = train_model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2,
    }