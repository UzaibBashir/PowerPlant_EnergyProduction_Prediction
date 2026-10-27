from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "models" / "model.pt"
DATA_PATH = PROJECT_DIR / "powerplant_data.csv"

#Model
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, inputs):
        return self.model(inputs)
