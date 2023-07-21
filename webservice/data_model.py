from pydantic import BaseModel


class TaxiRide(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float
    fare_amount: float
    total_amount: float


class TaxiRidePrediction(TaxiRide):
    prediction: float
