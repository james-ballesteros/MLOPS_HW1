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

@asset(group_name=asset_group_name, compute_kind="pandas")
def sample_new_data():
    """
    Provide a sample dataset for making predictions.

    Returns:
        pd.DataFrame: Sample new data for prediction.
    """
    # Sample new data
    data = {
        "CO(GT)": [0.4569845515154895, 0.4702654125698419, 0.4699875462156584],
        "PT08.S1(CO)": [0.9398556321215549, 0.9404656548512564, 0.9581202698751227],
        "NMHC(GT)": [2.0654865655141884, 2.2984562154862575, 2.2995541211544115],
        "C6H6(GT)": [0.2265465123221584, 0.2415255841412112, 0.2658441521212125],
        "PT08.S2(NMHC)": [0.4214556158791231, 0.4359874512154415, 0.4702125488187941],
        "NOx(GT)": [-0.0120226544464471, -0.0113945110121187, -0.0099987878454511],
        "PT08.S3(NOx)": [0.7188741154445056, 0.7845126854852165, 0.8954154112898122],
        "NO2(GT)": [0.4047487273862507, 0.4278741233659651, 0.4587454133952128],
        "PT08.S4(NO2)": [0.5871641709647203, 0.6198712532587941, 0.7112598412148558],
        "PT08.S5(O3)": [0.5949947793741827, 0.6215410254515845, 0.6994541479941121],
        "T": [0.078999725, 0.083215448, 0.092659841],
        "RH": [0.1789476103332271, 0.1821121514188952, 0.2125689874225485],
        "AH": [0.176064487, 0.189985544, 0.209545541],
        "CO_Norm": [2.0784524509213695, 2.4725988987452187, 2.8455245709213695],
    }
    return pd.DataFrame(data)

@asset(group_name=asset_group_name, compute_kind="scikit-learn")
def make_predictions(train_model, sample_new_data):
    """
    Use the trained model to make predictions on new sample data.

    Args:
        train_model (RandomForestRegressor): Trained regression model.
        sample_new_data (pd.DataFrame): New data for which predictions need to be made.

    Returns:
        pd.DataFrame: Predictions alongside the input data.
    """
    # Ensure the sample data is not empty
    if sample_new_data.empty:
        raise ValueError("The sample data is empty. Please provide valid input.")

    # Make predictions
    predictions = train_model.predict(sample_new_data)

    # Combine the input data with predictions
    prediction_results = sample_new_data.copy()
    prediction_results["Predictions"] = predictions

    return prediction_results