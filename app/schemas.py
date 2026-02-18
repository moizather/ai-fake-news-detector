from pydantic import BaseModel

class NewsInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    prediction: str
