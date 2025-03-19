#Definition of the prediction functions using the differement models
#AR, MA, ARMA, SARIMAX

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from app.database import parse_energy_data

def forecast_ar(xls_file, steps=10):
    return _forecast_generic(xls_file, steps, order=(1, 0, 0))

def forecast_ma(xls_file, steps=10):
    return _forecast_generic(xls_file, steps, order=(0, 0, 1))

def forecast_arma(xls_file, steps=10):
    return _forecast_generic(xls_file, steps, order=(1, 0, 1))

def forecast_sarimax(xls_file, steps=10):
    return _forecast_generic(xls_file, steps, sarimax=True)

#Generic prediction function taking in parameter the model of prediction
def _forecast_generic(xls_file, steps, order=None, sarimax=False):
    try:
        df = parse_energy_data(xls_file)

        if df.empty:
            print("No data available!")
            return []

        #Setting the field of DateTime as the index of the dataframe
        df.set_index("DateTime", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('15min')

        time_series = df['Consommation'].dropna() #We select all the values not null of the consommation field 

        if len(time_series) < 2:
            print("Not enough data for forecasting!")
            return []

        #Choice of the model
        if sarimax:
            model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 96))
        else:
            model = ARIMA(time_series, order=order)

        #Predictions
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=steps)

        #Generates a sequence of future timestamps with a 15-minute interval.
        last_date = df.index.max() 
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="15min")[1:]

        #Creates a DataFrame and store the forecasted results inside it
        forecast_df = pd.DataFrame({"DateTime": future_dates, "Forecast": predictions})
        forecast_df["DateTime"] = forecast_df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_df["Forecast"] = forecast_df["Forecast"].apply(lambda x: f"{x:.2f} kWh")

        #Return the dataframe in a JSON format
        return forecast_df.to_dict(orient="records")

    except Exception as e:
        print(f"Error in forecast(): {e}")
        return []
