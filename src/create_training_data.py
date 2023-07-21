import os
import pandas as pd
from typing import Optional

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

def create(months: list[int], year: int):
    df = None
    save_path = "./data/green_tripdata_training.parquet"
    for month in months:
        file_name = f"green_tripdata_{year}-{month:02d}.parquet"
        extract_data(file_name)
        df = load_data(file_name, df2=df)
    
    print(f"Writing to training {save_path}")
    df.to_parquet(f"{save_path}")

if __name__ == "__main__":
    months = [1,2,3]
    year = 2021
    create(months, year)
