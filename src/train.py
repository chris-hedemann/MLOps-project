import os
import argparse
import pandas as pd
import mlflow
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Optional

def run(year, months, cml_run, local, model_name="mlops-project"):
    if local:
        load_dotenv()
        MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI")
        SA_KEY= os.getenv("SA_KEY")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY
        
    else:
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        GOOGLE_APPLICATION_CREDENTIALS = "./credentials.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
    
    features = ["PULocationID", 
                    "DOLocationID", 
                    "trip_distance", 
                    "fare_amount", 
                    "total_amount"]
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

    df = None
    for month in months:
        file_name = f"green_tripdata_{year}-{month:02d}.parquet"

        extract_data(file_name)
        df = load_data(file_name, df2=df)

    df = calculate_trip_duration_in_minutes(df)
    X_train, X_test, y_train, y_test = split_data(df)


    rmse_train, rmse_test, model_version, model_name = \
    train_model(model_name,
            X_train,
            X_test, 
            y_train, 
            y_test,
            tags,
            MLFLOW_TRACKING_URI)
    
    if cml_run:
        with open("metrics.txt", "w") as f:
            f.write(f"Model: {model_name}, Version: {model_version}")
            f.write(f"RMSE on the Train Set: {rmse_train}")
            f.write(f"RMSE on the Test Set: {rmse_test}")

def extract_data(file_name: str): 
    if not os.path.exists(f"./data/{file_name}"):
        os.system(f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}")

def load_data(file_name: str,
        df2: Optional[pd.DataFrame] = None,
        ) -> pd.DataFrame:
    
    df = pd.read_parquet(f"./data/{file_name}")
    if df2 is None:
        df = pd.concat([df, df2], ignore_index=True)
    return df

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
        rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        rmse_train = mean_squared_error(
            y_train,
            lr.predict(X_train),
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

        return rmse_train, rmse_test, model_version, model_name
    
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
    months = [1,2]
    year = 2021

    run(year, months, cml_run, local)