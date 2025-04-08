import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from app.database import parse_energy_data  # Fonction pour lire les fichiers .xls
from sklearn.linear_model import LinearRegression


# 📂 Dossier où sauvegarder les prédictions
SAVE_DIR = "saved_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)  # Crée le dossier s'il n'existe pas


def forecast_ar(xls_file, steps=30000):
    return _forecast_generic(xls_file, steps, model_name="AR", order=(1, 0, 0))

def forecast_ma(xls_file, steps=30000):
    return _forecast_generic(xls_file, steps, model_name="MA", order=(0, 0, 1))

def forecast_arma(xls_file, steps=30000):
    return _forecast_generic(xls_file, steps, model_name="ARMA", order=(1, 0, 1))

def forecast_sarimax(xls_file, steps=1):
    return _forecast_generic(xls_file, steps, model_name="SARIMAX")

def forecast_linear(xls_file, steps=30000):
    return _forecast_generic(xls_file=xls_file, steps=steps, model_name="LINEAR")



def _forecast_generic(xls_file, steps, model_name, order=None):
    try:
        # Vérifie si le fichier CSV existe déjà
        year = os.path.basename(xls_file).split("_")[-1].split(".")[0]  # Extrait l'année
        csv_file = os.path.join(SAVE_DIR, f"forecast_{model_name}_{year}.csv")

        if os.path.exists(csv_file):
            print(f"Chargement des prédictions depuis {csv_file}...")
            return pd.read_csv(csv_file).to_dict(orient="records")

        # Si le fichier n'existe pas, exécuter la prédiction
        df = parse_energy_data(xls_file)
        if df.empty:
            print("Aucune donnée disponible !")
            return []

        # Préparation des données
        df = df.reset_index()
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.set_index("DateTime", inplace=True)
        df = df.sort_index()
        df = df.asfreq("15min")
        df.dropna(inplace=True)

        time_series = df['Consommation'].dropna()
        if len(time_series) < 2:
            print("⚠️ Pas assez de données pour faire une prévision !")
            return []

        # Sélection et entraînement du modèle
        match model_name:
            case "LINEAR":
                X = pd.DataFrame({
                    "Year": time_series.index.year,
                    "Month": time_series.index.month,
                    "Day": time_series.index.day,
                    "Weekday": time_series.index.weekday,
                    "Hour": time_series.index.hour,
                })
                y = time_series

                # Création d'une nouvelle DataFrame pour les étapes futures
                future_dates = pd.date_range(start=time_series.index[-1] + pd.Timedelta(minutes=15), periods=steps, freq="15min")
                X_future = pd.DataFrame({
                    "Year": future_dates.year,
                    "Month": future_dates.month,
                    "Day": future_dates.day,
                    "Weekday": future_dates.weekday,
                    "Hour": future_dates.hour,
                })

                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X_future)  # Prédictions pour les prochains "steps"
                predictions = pd.Series(predictions) # Conversion en objet panda.Series pour cohésion dans la suite du code
            case "SARIMAX" :
                model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 96))
                predictions = model.fit().forecast(steps=steps)
            case _ :
                model = ARIMA(time_series, order=order)
                predictions = model.fit().forecast(steps=steps)

        # print(f"type of predictions for model {model_name} : {type(predictions)}")    
        # predictions = predictions.flatten() # Pour s'assurer que les prédictions sont dans un tableau numpy 1D

        # Vérification de la longueur des prédictions et des dates futures
        if len(predictions) != steps:
            raise ValueError(f"Les longueurs des prédictions ({len(predictions)}) et des steps ({steps}) ne correspondent pas.")

        # Génération des dates futures
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="15min")[1:]

        # Vérification de la longueur des dates futures
        if len(future_dates) != len(predictions):
            raise ValueError(f"Les longueurs des dates futures ({len(future_dates)}) et des prédictions ({len(predictions)}) ne correspondent pas.")

        # Création du DataFrame avec les prédictions
        forecast_df = pd.DataFrame({"DateTime": future_dates, "Forecast": predictions})

        # Formatage des dates
        forecast_df["DateTime"] = forecast_df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Formatage des prédictions
        forecast_df["Forecast"] = forecast_df["Forecast"].apply(lambda x: f"{x:.2f} kWh")

        # Sauvegarde des prédictions dans un fichier CSV
        forecast_df.to_csv(csv_file, index=False)
        print(f"Prédictions sauvegardées dans {csv_file}")

        return forecast_df.to_dict(orient="records")

    except Exception as e:
        print(f"Erreur dans _forecast_generic(): {e}")
        return []
