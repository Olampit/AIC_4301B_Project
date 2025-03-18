from fastapi import FastAPI
from app.model import forecast_ar, forecast_ma, forecast_arma, forecast_sarimax

app = FastAPI()

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

        model_map = {
            "AR": forecast_ar,
            "MA": forecast_ma,
            "ARMA": forecast_arma,
            "SARIMAX": forecast_sarimax,
        }

        if model.upper() not in model_map:
            return {"error": "Invalid model type. Use AR, MA, ARMA, or SARIMAX."}

        result = model_map[model.upper()](xls_file=xls_file)

        return {"forecast": result}

    except Exception as e:
        return {"error": str(e)}
