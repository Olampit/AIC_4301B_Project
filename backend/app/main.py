from fastapi import FastAPI
from app.model import forecast

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the forecasting API"}


@app.get("/predict/{year}")
def predict(year: int):
    """
    Predicts the consumption for the specified year.
    :param year: the year you want the prediction to be done on
    :return: prediction results
    """
    try:
        xls_file = f"data/conso_mix_RTE_{year}.xls" 

        result = forecast(xls_file=xls_file)

        return {"forecast": result}

    except Exception as e:
        return {"error": str(e)}
