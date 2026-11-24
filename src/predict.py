import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src import PROJECT_ROOT
from src.preprocessing import CATEGORY_OPTIONS, FEATURES, NUMERIC_FEATURES

MODEL_PATH = PROJECT_ROOT / "models" / "modelo.pkl"

#: Rangos permitidos (mínimo, máximo) para cada variable numérica.
NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0.0, 500.0),
    "TotalCharges": (0.0, 60000.0),
}

LABELS = {0: "No Churn", 1: "Churn"}

#: Casos de ejemplo para pruebas y demostraciones en vivo.
EJEMPLOS = {
    "Cliente con alto riesgo de abandono": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.70,
        "TotalCharges": 191.40,
    },
    "Cliente estable (bajo riesgo)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.50,
        "TotalCharges": 3930.00,
    },
}


def cargar_modelo(ruta: Path = MODEL_PATH):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")
    return joblib.load(ruta)


def validar_entrada(datos: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(datos, Mapping):
        raise ValueError("La entrada debe ser un diccionario de features.")

    faltantes = [f for f in FEATURES if f not in datos]
    if faltantes:
        raise ValueError(f"Faltan campos obligatorios: {', '.join(faltantes)}")

    errores = []
    registro = {}
    for feature in NUMERIC_FEATURES:
        valor = datos[feature]
        if valor is None or isinstance(valor, bool):
            errores.append(f"'{feature}' debe ser un número (recibido: {valor!r}).")
            continue
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            errores.append(f"'{feature}' debe ser un número (recibido: {valor!r}).")
            continue
        if math.isnan(numero) or math.isinf(numero):
            errores.append(f"'{feature}' no puede ser nulo ni infinito.")
            continue
        minimo, maximo = NUMERIC_RANGES[feature]
        if not minimo <= numero <= maximo:
            errores.append(
                f"'{feature}' = {numero} está fuera del rango [{minimo}, {maximo}]."
            )
            continue
        registro[feature] = numero

    for feature, opciones in CATEGORY_OPTIONS.items():
        valor = datos[feature]
        if valor not in opciones:
            errores.append(
                f"'{feature}' = {valor!r} no es válido. Opciones: {opciones}."
            )
            continue
        registro[feature] = valor

    if "SeniorCitizen" in registro and registro["SeniorCitizen"] not in (0.0, 1.0):
        errores.append("'SeniorCitizen' debe ser 0 o 1.")

    if errores:
        raise ValueError(" ".join(errores))

    registro["SeniorCitizen"] = int(registro["SeniorCitizen"])
    registro["tenure"] = int(registro["tenure"])
    return pd.DataFrame([registro], columns=FEATURES)


def predecir(modelo, datos: Mapping[str, Any], umbral: float = 0.5) -> dict:
    if not 0.0 < umbral < 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    df = validar_entrada(datos)
    probabilidad = float(modelo.predict_proba(df)[0, 1])
    clase = int(probabilidad >= umbral)
    return {"clase": clase, "etiqueta": LABELS[clase], "probabilidad": probabilidad}
