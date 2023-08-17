import pickle

with open("./models/sentiment_analysis_pretrained.pkl", "rb") as model_file:
    sa_model = pickle.load(model_file)

def get_sentiment(text: str) -> list:
    prediction, _ = sa_model.predict([text])
    sentiment = None
    if prediction[0] == 0:
        sentiment = "Negative"
    if prediction[0] == 1:
        sentiment = "Neutral"
    if prediction[0] == 2:
        sentiment = "Positive"
    return sentiment