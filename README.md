# Predicción de Abandono de Clientes (Customer Churn) con XGBoost

## Descripción

Este proyecto desarrolla un sistema completo de *Machine Learning* para **predecir si un cliente de una empresa de telecomunicaciones abandonará el servicio** (*churn*). A partir de su perfil demográfico, los servicios contratados y la información de facturación, el modelo estima la probabilidad de abandono y la clasifica como `Churn` o `No Churn`.

El abandono de clientes es uno de los problemas más costosos para las empresas de servicios por suscripción: adquirir un cliente nuevo suele costar entre cinco y siete veces más que retener uno existente. Identificar con anticipación a los clientes en riesgo permite a las áreas de marketing y servicio al cliente enfocar sus campañas de retención (descuentos, migración a contratos anuales, soporte técnico prioritario) en quienes realmente lo necesitan, optimizando el presupuesto y reduciendo la pérdida de ingresos recurrentes.

El proyecto incluye el análisis exploratorio, el preprocesamiento reproducible, el entrenamiento y ajuste de hiperparámetros de un modelo XGBoost, su evaluación con métricas estándar y una aplicación web interactiva en **Streamlit** lista para desplegarse en Streamlit Cloud.

## Demostración

[Enlace a la aplicación desplegada](URL_AQUI)

La aplicación incluye dos **casos de ejemplo predefinidos** en la barra lateral (un cliente con alto riesgo de abandono y un cliente estable) para realizar una demostración en vivo en menos de un minuto.

## Algoritmo utilizado

Se utiliza **Gradient Boosting con XGBoost** (`XGBClassifier` de la librería `xgboost`), entrenado desde cero sin *transfer learning*.

**Justificación técnica:**

- **Datos tabulares heterogéneos:** el dataset combina variables numéricas y categóricas; los ensambles de árboles potenciados por gradiente son el estado del arte para este tipo de datos.
- **Relaciones no lineales e interacciones:** XGBoost captura automáticamente interacciones como *contrato mes a mes + fibra óptica + sin soporte técnico*, que un modelo lineal no modela bien.
- **Regularización integrada** (`reg_lambda`, `gamma`, `min_child_weight`, `subsample`, `colsample_bytree`) que reduce el sobreajuste en un dataset de tamaño medio.
- **Robustez** frente a outliers y escalas distintas, entrenamiento rápido (`tree_method="hist"`) e **interpretabilidad** mediante la importancia de características.

El modelo se entrena dentro de un `Pipeline` de scikit-learn junto con el preprocesador (`ColumnTransformer`), y sus hiperparámetros se ajustan con `RandomizedSearchCV` (40 combinaciones, validación cruzada estratificada de 5 folds, métrica ROC-AUC). Mejores hiperparámetros: `n_estimators=300`, `learning_rate=0.02`, `max_depth=3`, `min_child_weight=7`, `subsample=0.8`, `colsample_bytree=0.5`, `gamma=0`, `reg_lambda=1`.

**Métricas reales obtenidas en el conjunto de prueba (20 %, 1,409 clientes):**

| Métrica | Valor |
|---|---|
| Accuracy | **0.8084** (80.84 %) |
| Precision (Churn) | 0.6806 |
| Recall (Churn) | 0.5241 |
| F1-Score (Churn) | 0.5921 |
| ROC-AUC | **0.8488** |
| ROC-AUC (validación cruzada, 5 folds) | 0.8506 |

El accuracy supera ampliamente el mínimo requerido del 70 %. Debido al desbalance de clases (26.5 % de churn), el recall es moderado; en la aplicación se puede reducir el **umbral de decisión** para detectar más clientes en riesgo a cambio de más falsos positivos.

Las variables más importantes para el modelo son: tipo de contrato *mes a mes*, ausencia de soporte técnico, ausencia de seguridad online, internet por fibra óptica, pago con cheque electrónico y antigüedad (`tenure`).

## Dataset

- **Fuente:** [Telco Customer Churn — IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) (también disponible en Kaggle).
- **Archivo:** `data/raw/telco_churn.csv`.
- **Tamaño:** **7,043 registros y 21 columnas** (19 features + identificador + variable objetivo).
- **Variable objetivo:** `Churn` (Yes = 1,869 clientes / No = 5,174 clientes).

**Features utilizadas (19):**

| Grupo | Variables |
|---|---|
| Demográficas | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Cuenta | `tenure` (meses de antigüedad), `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Facturación | `MonthlyCharges`, `TotalCharges` |
| Servicios | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Preprocesamiento** (`src/preprocessing.py`): se elimina `customerID`; `TotalCharges` se convierte a numérico y sus 11 valores vacíos (clientes con `tenure = 0`) se imputan con 0; las variables numéricas se escalan con `StandardScaler` y las categóricas se codifican con `OneHotEncoder`. El dataset limpio y las particiones train/test se guardan en `data/processed/`.

## Instalación local

### Requisitos

- Python 3.11 o superior (probado con Python 3.15; para Streamlit Cloud se recomienda 3.12 o 3.13).
- `pip` y, opcionalmente, `venv` para crear un entorno virtual.

### Pasos

```bash
git clone https://github.com/USUARIO/proyecto-xgboost-grupo7.git
cd proyecto-xgboost-grupo7

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
streamlit run app/app.py
```

La aplicación se abrirá en `http://localhost:8501`.

Para volver a entrenar el modelo, abre y ejecuta `notebooks/01_training.ipynb` (Jupyter o VS Code); el notebook regenera `data/processed/` y `models/modelo.pkl`.

## Uso

1. Ejecuta `streamlit run app/app.py` desde la raíz del proyecto (o abre la URL desplegada).
2. En la barra lateral, pulsa **"Cliente con alto riesgo de abandono"** o **"Cliente estable (bajo riesgo)"** para cargar un caso de ejemplo, o completa manualmente los campos de las tres secciones: *Perfil del cliente*, *Cuenta y facturación* y *Servicios contratados*.
3. (Opcional) Ajusta el **umbral de decisión** en la barra lateral (por defecto 0.50).
4. Pulsa **"🔮 Predecir abandono"**.
5. Revisa el resultado: la clase predicha (`Churn` / `No Churn`), la probabilidad de abandono con su barra de progreso y la interpretación del nivel de riesgo con una recomendación de negocio.
6. Usa **"Restablecer valores"** para volver a los valores por defecto.

También puedes usar el modelo desde Python:

```python
from src.predict import EJEMPLOS, cargar_modelo, predecir

modelo = cargar_modelo()
print(predecir(modelo, EJEMPLOS["Cliente con alto riesgo de abandono"]))
# {'clase': 1, 'etiqueta': 'Churn', 'probabilidad': 0.8...}
```

## Estructura del proyecto

```
proyecto-xgboost-grupo7/
├── app/
│   ├── app.py               # Aplicación web en Streamlit
│   ├── static/              # Recursos estáticos (reservado)
│   └── templates/           # Plantillas (reservado)
├── data/
│   ├── raw/
│   │   └── telco_churn.csv  # Dataset original de IBM
│   └── processed/
│       ├── telco_churn_clean.csv  # Dataset limpio
│       ├── train.csv              # Partición de entrenamiento (80 %)
│       └── test.csv               # Partición de prueba (20 %)
├── models/
│   └── modelo.pkl           # Pipeline serializado (preprocesador + XGBoost)
├── notebooks/
│   └── 01_training.ipynb    # EDA, entrenamiento y evaluación (ejecutado)
├── src/
│   ├── __init__.py          # Define la raíz del proyecto (PROJECT_ROOT)
│   ├── preprocessing.py     # Limpieza y ColumnTransformer
│   └── predict.py           # Carga del modelo, validación e inferencia
├── .streamlit/
│   └── config.toml          # Tema visual de la aplicación
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Todo el código usa rutas relativas a la raíz del proyecto mediante `pathlib.Path`, por lo que funciona igual en Windows y en Linux (Streamlit Cloud).

## Autor

Proyecto individual — **Daniel Andrés Lerzundy Torres**
Asignatura: Inteligencia Artificial I

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
