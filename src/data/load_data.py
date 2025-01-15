import pandas as pd
import os

def load_airquality_data(source_folder="data/03_Gold/-1_Dev", filename="AirQualityUCI.csv"):
    """
    Load the AirQualityUCI dataset from a specified source folder.

    Args:
        source_folder (str): Path to the folder containing the CSV file.
        filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = os.path.join(source_folder, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path, sep=";", decimal=",")

# Example usage
if __name__ == "__main__":
    airqual_df = load_airquality_data()
    print(airqual_df.head())