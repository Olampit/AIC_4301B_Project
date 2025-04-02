# AIC_4301B_Project
## 4301B project by Thomas PERNOT, Terry OLAMPI, Mickael TRAN and Margot UTRERA

This project is a forecasting platform that integrates different time series models, including AR, MA, ARMA, and SARIMAX. The platform is structured using microservices and containerized with Docker. It consists of:

- **Data Storage Service**: A database to store time series data.

- **Back-end Service**: A FastAPI-based API to process and forecast data.

- **Front-end Service**: A dashboard to visualize forecasts.

---

# Requirements : 

- Python 3.8+

- pip (included with python)

- Docker (for containerization)

---


## Done :

Created the API (working).

Finished data handling (for missing values).

Implemented the models (AR, MA, ARMA, SARIMAX).

Implemented a graphical view of the predictions.

Developped a front-end visualization.

Containerized the files with docker.


## To do :

Ajouter un outil de sélection de dates où l'utilisateur peut choisir la durée et le début des prédictions (et faire des valeurs de base cohérentes)

Implement the database service. => faire un vrai serveur de base de données

---

Overall :

We should modify the functions so that we do not have to refit the models and we only have to predict using them / will not be done because we would have to change the entire structure and when we tried it did not work.
We should modify the data storage to have it separated from the model part 
We should containerize every part of the project, not just the project as a whole ? 
We should orchestrate the containers i guess

---

# Contributors

## Thomas PERNOT, Terry OLAMPI, Mickael TRAN and Margot UTRERA
