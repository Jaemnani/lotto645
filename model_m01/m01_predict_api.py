from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
from m01_model import LSTMNet
import os

app = FastAPI()

class LottoInput(BaseModel):
    history: list[list[int]]  # one-hot 형태


# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

# torch.serialization.add_safe_globals({'LSTMNet': LSTMNet})

win_size = 64
model = LSTMNet(input_size=45, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load("m01_model.pt", map_location=device))
model.eval()

@app.post("/predict")
def predict(input: LottoInput):
    x = torch.tensor([input.history], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
        pred = sorted((pred[0].argsort()[-6:] + 1).numpy())
        pred = [int(num) for num in pred]
        return {"numbers": pred}
