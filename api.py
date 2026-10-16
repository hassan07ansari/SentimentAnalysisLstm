from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from predict import predict_sentiment

app = FastAPI(
    title="IMDB Sentiment Analysis Api",
    description="Predict movie review sentiment using Word2Vec and LSTM.",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    review : str = Field(
        min_length=1,
        max_length=10000,
        description="Movie review text"
    )


@app.get("/")
def home():
    return {"message": "IMDb Sentiment Analysis API is running"}




@app.post("/predict")
def predict(request: ReviewRequest):
    try:
        return predict_sentiment(request.review)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from e

    