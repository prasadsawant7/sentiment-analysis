from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentiment_analysis import get_sentiment

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/v1/test")
def index():
    return {"message": "Connection Successfull!"}

@app.post("/api/v1/sentiment-analysis")
async def send_text_to_analyse_sentiment(data: dict) -> dict:
    sentiment = get_sentiment(data["text"])
    response = {
        "text": data["text"],
        "sentiment": sentiment
    }
    return { "status": "ok", "response": response }