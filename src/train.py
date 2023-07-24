import os
import argparse
import pandas as pd
import mlflow
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run(year, months, cml_run, local, model_name="mlops-project"):
    ## Environmental variables
    if local:
        load_dotenv()
    else:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"
    
    MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")

    ## Set up meta information
    features = [
        "PULocationID", 
        "DOLocationID", 
        "trip_distance", 
        "fare_amount", 
        "total_amount"
        ]
    target = "duration"
    tags = {
        "model": "linear regression",
            "developer": "Chris Hedemann",
            "dataset": f"green-taxi",
            "year": year,
            "month": months,
            "features": features,
            "target": target
    }


    ## Prepare data
    df = pd.read_parquet('./data/green_tripdata_training.parquet')
    df = calculate_trip_duration_in_minutes(df)
    X_train, X_test, y_train, y_test = split_data(df)

    ## Train model
    rmse_train, rmse_test, model_version, model_name, ref_data = \
    train_model(model_name,
            X_train,
            X_test, 
            y_train, 
            y_test,
            tags,
            MLFLOW_TRACKING_URI)
    
    ## Write reference data to evidently
    ref_data = ref_data[features]
    ref_data.to_csv("gs://training-data-mlops-project/reference.csv", 
              index=False,
              storage_options={
                  'token': os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
                  }
                  )

    ## Write metrics to file
    if cml_run:
        with open("metrics.txt", "w") as f:
            f.write(f"Model: {model_name}, version {model_version}\n")
            f.write(f"––––––"*4 + "\n")
            f.write(f"RMSE on the Train Set: {rmse_train}\n")
            f.write(f"RMSE on the Test Set: {rmse_test}\n")

def calculate_trip_duration_in_minutes(df: pd.DataFrame):
    features = ["PULocationID", 
                "DOLocationID", 
                "trip_distance", 
                "fare_amount", 
                "total_amount"]
    target = "duration"

    df[target] = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[features + [target]]
    return df

def split_data(df: pd.DataFrame):
    df = df.copy()
    target = "duration"
    y=df[target]
    X=df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        random_state=42, 
        test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_model(model_name: str, 
                X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.Series, 
                y_test: pd.Series,
                tags: dict[str],
                MLFLOW_TRACKING_URI):

    # Set up the connection to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(model_name)

    with mlflow.start_run():

        # Train and evluate model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_train_pred = lr.predict(X_train)
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        rmse_train = mean_squared_error(
            y_train,
            y_train_pred,
            squared=False
            )

        # Log model
        mlflow.set_tags(tags)
        mlflow.log_metric("rmse", rmse_test)
        mlflow.sklearn.log_model(lr, "model")

        # Register and transition model to production
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        latest_versions = client.get_latest_versions(model_name)
        model_version = int(latest_versions[-1].version)
        new_stage = "Production"
        client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
        )

        ref_data = X_train.copy()
        ref_data['duration'] = y_train
        ref_data['prediction'] = y_train_pred

        return rmse_train, rmse_test, model_version, model_name, ref_data
    
# def update_reference(data: TaxiRide):
#     MONITORING_SERVICE_URI = os.getenv("MONITORING_SERVICE_URI")
#     try:
#         response = requests.post( 
#             f"{MONITORING_SERVICE_URI}/iterate/green_taxi_data", 
#             data=None,
#             headers={"content-type": "application/json"},
#         )
#     except requests.exceptions.ConnectionError as error:
#         print(f"Cannot reach a metrics application, error: {error}, data: {data}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cml_run", default=False, action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument(
        "--local", default=False, action=argparse.BooleanOptionalAction, required=True
    )
    args = parser.parse_args()

    cml_run = args.cml_run
    local = args.local
    months = [2,]
    year = 2022

    run(year, months, cml_run, local)