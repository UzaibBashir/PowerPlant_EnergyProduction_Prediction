from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "model.pt"

#Load Model
model = ANNModel()