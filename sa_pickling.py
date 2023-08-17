from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import emoji
import pickle

cuda_available = torch.cuda.is_available()

model = ClassificationModel(
    model_type="bertweet",
    model_name="finiteautomata/bertweet-base-sentiment-analysis",
    use_cuda=False,
)

with open("./models/sentiment_analysis_pretrained.pkl", "wb") as model_file:
    pickle.dump(model, model_file)