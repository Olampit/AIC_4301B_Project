import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from app.database import parse_energy_data

def forecast(xls_file="../data/conso_mix_RTE_2025.xls", steps=10):
    try:
        df = parse_energy_data(xls_file)

        time_series = df['Consommation']

        print("Time series data:", time_series.head())  # Debugging line

        time_series = time_series.dropna()

        if len(time_series) < 2:
            print("Not enough data for forecasting!")
            return []

        model = ARIMA(time_series, order=(2, 1, 2))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=steps)

        return predictions.tolist()
    except Exception as e:
        print(f"Error in forecast() : {e}")
        return []
