import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from app.database import parse_energy_data

def forecast(xls_file="../data/conso_mix_RTE_2025.xls", steps=365):
    try:
        df = parse_energy_data(xls_file)

        if df.empty:
            print("No data available!")
            return pd.DataFrame(columns=["DateTime", "Forecast"])

        df.set_index("DateTime", inplace=True)

        time_series = df['Consommation'].dropna()

        if len(time_series) < 2:
            print("Not enough data for forecasting!")
            return pd.DataFrame(columns=["DateTime", "Forecast"])

        model = ARIMA(time_series, order=(2, 1, 2)) 
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=steps)

        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="h")[1:]

        forecast_df = pd.DataFrame({"DateTime": future_dates, "Forecast": predictions})
        forecast_df["Forecast"] = forecast_df["Forecast"].apply(lambda x: f"{x:.2f} kWh")

        return forecast_df
    except Exception as e:
        print(f"Error in forecast(): {e}")
        return pd.DataFrame(columns=["DateTime", "Forecast"])
