# Write down here.
import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    # bhk: int
    # size: int
    # floor: str
    # area_type: str
    # city: str
    # furnishing_status: str
    # tenant_preferred: str
    # bathroom: int
    # point_of_contact: str
    datetime: str
    season: str
    holiday: int
    workingday: int
    weather: str
    temp: float
    atemp: float
    humidity: int
    windspeed: float


bento_model = bentoml.sklearn.get("bike_sharing:latest")
model_runner = bento_model.to_runner()

svc = bentoml.Service("bike_sharing_regressor", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=Features), output=NumpyNdarray())
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(input_df)
    return np.expm1(log_pred)