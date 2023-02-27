import geopandas as gpd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


class GeoFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, get_poi_data, get_zone_data):
        super().__init__()

        df_zone = get_zone_data()
        df_poi = get_poi_data()

        df_poi = gpd.sjoin(
            df_poi, df_zone[["geometry"]], predicate="within", how="left"
        )
        df_poi = df_poi.rename(columns={"index_right": "LocationID"})
        df_poi["LocationID"] = df_poi["LocationID"].astype("Int64")

        df_zone["POI_count"] = df_poi.groupby("LocationID").size()
        df_zone["POI_count"] = df_zone["POI_count"].fillna(0)

        df_zone["Hotel_count"] = (
            df_poi[df_poi["FACI_DOM"] == "Hotel/Motel"].groupby("LocationID").size()
        )
        df_zone["Hotel_count"] = df_zone["Hotel_count"].fillna(0)

        df_zone["Residential_count"] = (
            df_poi[df_poi["FACILITY_T"] == "Residential"].groupby("LocationID").size()
        )
        df_zone["Residential_count"] = df_zone["Residential_count"].fillna(0)

        df_zone["has_airport"] = (
            df_poi[df_poi["FACI_DOM"] == "Airport"].groupby("LocationID").size()
        )
        df_zone["has_airport"] = df_zone["has_airport"].fillna(0) > 1

        self.df_zone = df_zone

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        columns_to_add = [
            "Borough",
            "Zone",
            "service_zone",
            "Shape_Leng",
            "Shape_Area",
            "POI_count",
            "Hotel_count",
            "Residential_count",
            "has_airport",
        ]
        X = X.merge(
            self.df_zone[columns_to_add].add_prefix("PU"),
            left_on="PULocationID",
            right_index=True,
            how="left",
        )
        X = X.merge(
            self.df_zone[columns_to_add].add_prefix("DO"),
            left_on="DOLocationID",
            right_index=True,
            how="left",
        )
        return X


class WeatherFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, get_weather_data):
        super().__init__()

        df_precipitation = get_weather_data()
        self.df_precipitation = df_precipitation

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        columns_to_add = ["temp", "prcp"]
        X = X.merge(
            self.df_precipitation[columns_to_add],
            left_on=X.tpep_pickup_datetime.dt.floor("h"),
            right_index=True,
            how="left",
        )
        return X


FEATURES = [
    # 'VendorID',
    # 'tpep_pickup_datetime',
    # 'tpep_dropoff_datetime',
    "passenger_count",
    #    'PULocationID',
    #    'DOLocationID',
    #    'PUBorough',
    #    'PUZone',
    #    'PUservice_zone',
    "PUShape_Leng",
    "PUShape_Area",
    "PUPOI_count",
    "PUHotel_count",
    "PUResidential_count",
    "PUhas_airport",
    #    'DOBorough',
    #    'DOZone',
    #    'DOservice_zone',
    "DOShape_Leng",
    "DOShape_Area",
    "DOPOI_count",
    "DOHotel_count",
    "DOResidential_count",
    "DOhas_airport",
    "temp",
    "prcp",
]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select some features."""

    def __init__(self, features):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        return X[self.features]


def get_estimator(get_poi_data, get_weather_data, get_zone_data):
    geographic_feature_extractor = GeoFeatureExtractor(get_poi_data, get_zone_data)
    weather_feature_extractor = WeatherFeatureExtractor(get_weather_data)
    features_selector = FeatureSelector(features=FEATURES)

    # Replace NaN values
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")

    regressor = LinearRegression()

    pipe = make_pipeline(
        geographic_feature_extractor,
        weather_feature_extractor,
        features_selector,
        imputer,
        regressor,
    )

    return pipe
