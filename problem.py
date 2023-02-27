import json
import os

import geopandas as gpd
import pandas as pd
import rampwf as rw
from rampwf.utils.importing import import_module_from_source
from rampwf.workflows import SKLearnPipeline
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.utils import _safe_indexing

# --------------------------------------------------
#
# Challenge title

problem_title = "Taxi Fare Prediction"


# --------------------------------------------------
#
# Select Prediction type

Predictions = rw.prediction_types.make_regression()


# --------------------------------------------------
#
# Select Workflow


class EstimatorAdditionalData(SKLearnPipeline):
    """
    RAMP workflow for the Taxi prediction challenge. Its intended use is for
    training different models.

    Submissions need to contain one file: estimator.py, with the following
    requirements:
        - estimator.py : submitted file
            - function get_estimator : create a estimator
                The function is called with 3 arguments for the 3 additionnal dataset :
                    model = get_estimator(get_poi_data, get_weather_data, get_zone_data)
    """

    def __init__(self):
        super().__init__()

    def train_submission(self, module_path, X, y, train_idx=None):
        """Train the estimator of a given submission.

        Parameters
        ----------
        module_path : str
            The path to the submission where `filename` is located.
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The data matrix.
        y : array-like of shape (n_samples,)
            The target vector.
        train_idx : array-like of shape (n_training_samples,), default=None
            The training indices. By default, the full dataset will be used
            to train the model. If an array is provided, `X` and `y` will be
            subsampled using these indices.

        Returns
        -------
        estimator : estimator object
            The scikit-learn fitted on (`X`, `y`).
        """
        train_idx = slice(None, None, None) if train_idx is None else train_idx
        submission_module = import_module_from_source(
            os.path.join(module_path, self.filename),
            os.path.splitext(self.filename)[0],  # keep the module name only
            sanitize=True,
        )
        estimator = submission_module.get_estimator(
            get_poi_data, get_weather_data, get_zone_data
        )
        X_train = _safe_indexing(X, train_idx)
        y_train = _safe_indexing(y, train_idx)
        return estimator.fit(X_train, y_train)


workflow = EstimatorAdditionalData()


# --------------------------------------------------
#
# Define the score types

score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
    rw.score_types.NormalizedRMSE(name="normalized_rmse", precision=3),
]

# --------------------------------------------------
# CV scheme


def get_cv(X, y):
    cv = ShuffleSplit(
        n_splits=2,
        train_size=0.8,
        random_state=42,
    )
    return cv.split(X, y)


# --------------------------------------------------
# Get Data


def _read_data(path):
    data_file = os.path.join(path, "data", "yellow_tripdata_2022-05.parquet")
    df_parquet = pd.read_parquet(data_file, engine="pyarrow")
    # Filter year and month (erratic values)
    df_parquet = df_parquet.loc[
        (df_parquet.tpep_pickup_datetime.dt.year == 2022)
        & (df_parquet.tpep_pickup_datetime.dt.month == 5)
    ]
    # Miles to Kilometer
    df_parquet.trip_distance = df_parquet.trip_distance * 1.609344
    # Remove error in distance
    df_parquet = df_parquet.loc[df_parquet.trip_distance < 1000]
    # Remove error in price
    df_parquet = df_parquet.loc[
        (df_parquet.total_amount > 1) & (df_parquet.total_amount < 2000)
    ]
    # Clean Columns linked to y
    remove_col = [
        "payment_type",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "trip_distance",
        "RatecodeID",
        "store_and_fwd_flag",
    ]
    df_parquet = df_parquet.drop(remove_col, axis=1)
    # Split X,y
    X = df_parquet.loc[:, df_parquet.columns != "total_amount"]
    y = df_parquet.total_amount
    # Split Train, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=29
    )
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


def get_weather_data():
    data_file = os.path.join("data", "Precipitations_May_2022.csv")
    df_weather = pd.read_csv(data_file)
    # Change type of columns to datetime
    df_weather.time = pd.to_datetime(df_weather.time)
    df_weather.set_index("time", inplace=True)
    # Select columns to keep
    cols = ["temp", "prcp"]
    df_weather = df_weather[cols]
    return df_weather


def get_train_data(path="."):
    X_train, _, y_train, _ = _read_data(path)
    return X_train, y_train


def get_test_data(path="."):
    _, X_test, _, y_test = _read_data(path)
    return X_test, y_test


def get_zone_data():
    df_zone = pd.read_csv("data/taxi_zone_lookup.csv").set_index("LocationID")
    shapefile = gpd.read_file("data/taxi_zones/taxi_zones.shp")[
        ["OBJECTID", "Shape_Leng", "Shape_Area", "geometry"]
    ]
    shapefile = shapefile.rename(columns={"OBJECTID": "LocationID"})
    shapefile = shapefile.set_index("LocationID")
    df_zone = df_zone.merge(shapefile, left_index=True, right_index=True, how="left")
    df_zone = gpd.GeoDataFrame(df_zone, geometry="geometry").set_crs(shapefile.crs)

    return df_zone


def get_poi_data():
    columns = ["FACI_DOM", "BOROUGH", "FACILITY_T", "NAME", "geometry"]
    df_poi = gpd.read_file("data/Point_Of_Interest/Point_Of_Interest.shp")[columns]

    with open("data/Point_Of_Interest/attributes.json") as f:
        dico_attributes = json.load(f)

    def decode_row(row):
        if pd.notna(row["BOROUGH"]):
            code_borough = str(row["BOROUGH"])
            row["BOROUGH"] = dico_attributes["BOROUGH"][code_borough]
        code_facility_t = str(row["FACILITY_T"])
        code_faci_dom = str(row["FACI_DOM"])
        row["FACILITY_T"] = dico_attributes["FACILITY_T"][code_facility_t]
        row["FACI_DOM"] = dico_attributes["FACI_DOM"][code_facility_t][code_faci_dom]
        return row

    df_poi = df_poi.apply(decode_row, axis=1)

    return df_poi


if __name__ == "__main__":
    import rampwf

    os.environ["RAMP_TEST_MODE"] = "1"
    rampwf.utils.testing.assert_submission()
