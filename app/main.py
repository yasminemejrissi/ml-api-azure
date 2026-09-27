from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Charger le modèle UNE fois, au démarrage 
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
model = joblib.load(MODEL_PATH)

# 2. Créer l'API (le serveur)
app = FastAPI(title="API de detection de faux billets")

# 3. Le "menu" : ce qu'une commande doit contenir
class Billet(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

# 4. Page d'accueil (pour vérifier que l'API est en vie)
@app.get("/")
def accueil():
    return {"message": "API en ligne. Allez sur /docs pour tester."}

# 5. L'endpoint /predict : recevoir un billet, renvoyer la prédiction
@app.post("/predict")
def predict(billet: Billet):
    donnees = pd.DataFrame([billet.model_dump()])
    prediction = int(model.predict(donnees)[0])
    proba_faux = float(model.predict_proba(donnees)[0][1])
    return {
        "prediction": prediction,
        "resultat": "faux billet" if prediction == 1 else "vrai billet",
        "probabilite_faux": round(proba_faux, 3),
    }