from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model import forecast_ar, forecast_ma, forecast_arma, forecast_sarimax
import os
import pandas as pd

app = FastAPI()

# Active CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

# 📂 Répertoire où les prédictions seront enregistrées
SAVE_DIR = "saved_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the forecasting API"}

@app.get("/predict/{model}/{year}")
def predict(model: str, year: int):
    """
    Predicts the consumption for the specified year using a given model.
    :param model: the model type (AR, MA, ARMA, SARIMAX)
    :param year: the year for which prediction is needed
    :return: prediction results
    """
    try:
        xls_file = f"data/conso_mix_RTE_{year}.xls"
        csv_file = os.path.join(SAVE_DIR, f"forecast_{model.upper()}_{year}.csv")

        print(f"🔍 Vérification du fichier : {csv_file}")

        # 🔄 Si le fichier CSV existe, on le charge directement
        if os.path.exists(csv_file):
            print(f"📥 Chargement des prédictions depuis {csv_file}...")
            return {"forecast": pd.read_csv(csv_file).to_dict(orient="records")}

        model_map = {
            "AR": forecast_ar,
            "MA": forecast_ma,
            "ARMA": forecast_arma,
            "SARIMAX": forecast_sarimax,
        }

        if model.upper() not in model_map:
            return {"error": "Invalid model type. Use AR, MA, ARMA, or SARIMAX."}

        # 📈 Calcul de la prédiction et sauvegarde
        result = model_map[model.upper()](xls_file=xls_file)
        pd.DataFrame(result).to_csv(csv_file, index=False)
        print(f"✅ Prédictions sauvegardées dans {csv_file}")

        return {"forecast": result}

    except Exception as e:
        return {"error": str(e)}
