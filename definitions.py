from dagster import Definitions, load_assets_from_modules
from src import assets

# Load all assets from the `assets.py` module
all_assets = load_assets_from_modules([assets])

# Define resources and assets for Dagster
defs = Definitions(
    assets=all_assets,
    resources={
        "local_csv_io_manager": assets.local_csv_io_manager
    },
)
