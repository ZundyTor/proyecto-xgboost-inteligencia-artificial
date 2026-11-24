import base64
import html
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st

# Permite importar el paquete ``src`` sin importar desde dónde se lance la app.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (  # noqa: E402
    EJEMPLOS, MODEL_PATH, NUMERIC_RANGES, cargar_modelo, predecir, validar_entrada,
)
from src.preprocessing import CATEGORY_OPTIONS, FEATURES, NUMERIC_FEATURES  # noqa: E402

st.set_page_config(
    page_title="ChurnGuard Analytics | Predictor XGBoost",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Textos y constantes de presentación
# --------------------------------------------------------------------------- #
TRADUCCION = {
    "Yes": "Sí",
    "No": "No",
    "Female": "Femenino",
    "Male": "Masculino",
    "No phone service": "Sin servicio telefónico",
    "No internet service": "Sin servicio de internet",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)",
}

ETIQUETAS = {
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor (65+)",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "tenure": "Antigüedad",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Facturación digital",
    "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual",
    "TotalCharges": "Cargos totales",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Múltiples líneas",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad online",
    "OnlineBackup": "Respaldo online",
    "DeviceProtection": "Protección de dispositivo",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming películas",
}

PASOS = {
    1: ("Perfil del cliente", ["gender", "SeniorCitizen", "Partner", "Dependents"]),
    2: ("Cuenta y facturación", [
        "tenure", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
    ]),
    3: ("Servicios contratados", [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]),
}

VALORES_POR_DEFECTO = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check", "MonthlyCharges": 60.0, "TotalCharges": 1440.0,
}

# Tasas históricas de abandono calculadas sobre el dataset (data/processed).
TASA_CHURN_GLOBAL = 26.5
MEDIA_CARGO_MENSUAL = 64.76
TASA_CONTRATO = {"Month-to-month": 42.7, "One year": 11.3, "Two year": 2.8}
TASA_PAGO = {
    "Electronic check": 45.3, "Mailed check": 19.1,
    "Bank transfer (automatic)": 16.7, "Credit card (automatic)": 15.2,
}
CAPTION_CONTRATO = {
    "Month-to-month": "Mayor riesgo",
    "One year": "Riesgo medio",
    "Two year": "Alta retención",
}

METRICAS_MODELO = {
    "ROC-AUC (test)": "0.849",
    "Accuracy (test)": "80.8 %",
    "Espacio de variables": "19 features",
    "Validación cruzada": "5-fold estratificada",
}

COLORES = {
    "error": ("#ba1a1a", "#ffdad6", "#93000a"),
    "warning": ("#b45309", "#fef3c7", "#78350f"),
    "info": ("#0051d5", "#dbe1ff", "#00174b"),
    "success": ("#069669", "#d1fae5", "#005137"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300..600,0..1,0&display=swap');

:root {
  --surface: #f8f9ff; --card: #ffffff; --low: #eff4ff; --container: #e5eeff;
  --high: #dce9ff; --on-surface: #0b1c30; --muted: #45464d; --outline: #c6c6cd;
  --secondary: #0051d5; --error: #ba1a1a; --error-c: #ffdad6; --on-error-c: #93000a;
  --ok: #069669; --ok-c: #85f8c4; --on-ok-c: #002114;
}
.stApp { background: var(--surface); }
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.msym) {
  font-family: 'Inter', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .cg-title, .cg-h2, .cg-brand {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.msym {
  font-family: 'Material Symbols Outlined'; font-weight: normal; font-style: normal;
  line-height: 1; letter-spacing: normal; text-transform: none; white-space: nowrap;
  display: inline-block; direction: ltr; -webkit-font-smoothing: antialiased;
  font-size: 18px; vertical-align: middle;
}
[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1120px; padding-top: 1rem; padding-bottom: 1.5rem; }

/* ---------- Barra superior ---------- */
.st-key-topbar {
  position: sticky; top: .5rem; z-index: 999;
  background: rgba(248,249,255,.92); backdrop-filter: blur(16px);
  box-shadow: 0 1px 8px rgba(0,0,0,.06); border-radius: 12px; padding: .5rem 1rem;
}
.cg-brandrow { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.cg-logo {
  width: 32px; height: 32px; border-radius: 8px; background: #000; color: #85f8c4;
  display: flex; align-items: center; justify-content: center;
}
.cg-brand { font-size: 16px; font-weight: 700; color: var(--on-surface); letter-spacing: -.01em; }
.cg-tag {
  font-size: 10px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  background: var(--high); color: var(--muted); padding: 2px 6px; border-radius: 4px;
  border: 1px solid var(--outline);
}
.cg-live {
  display: inline-flex; align-items: center; gap: 6px; font-size: 10.5px; font-weight: 600;
  letter-spacing: .05em; text-transform: uppercase; background: var(--container);
  color: var(--on-surface); padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(0,81,213,.2); white-space: nowrap;
}
.cg-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--secondary);
  animation: cgpulse 1.6s infinite; }
@keyframes cgpulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
.st-key-caso [data-baseweb="select"] > div { background: #fff; border: 1px solid var(--outline); }

/* ---------- Encabezado ---------- */
.cg-eyebrow { display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase; color: var(--secondary); }
.cg-eyebrow .sep { width: 4px; height: 4px; border-radius: 50%; background: var(--outline); }
.cg-eyebrow .muted { color: var(--muted); font-weight: 500; text-transform: none; letter-spacing: 0; }
.cg-title { font-size: 28px; line-height: 36px; font-weight: 700; letter-spacing: -.02em;
  color: var(--on-surface); margin: 4px 0; }
.cg-sub { font-size: 14px; color: var(--muted); margin: 0; }
.st-key-params button { background: #fff; border: none; box-shadow: 0 1px 3px rgba(11,28,48,.12); }
.cg-flyhead { display: flex; justify-content: space-between; align-items: center; }
.cg-flyhead b { font-size: 15px; }
.cg-kv { background: var(--low); border-radius: 8px; padding: 8px 10px; font-size: 11.5px; color: var(--muted); }
.cg-kv div { display: flex; justify-content: space-between; padding: 2px 0; }
.cg-kv b { color: var(--on-surface); }

/* ---------- Tarjetas ---------- */
.st-key-stepper, .st-key-form_card, .st-key-panel {
  background: var(--card); border-radius: 14px; box-shadow: 0 1px 3px rgba(11,28,48,.07);
}
.st-key-stepper { padding: 1rem; }
.st-key-form_card, .st-key-panel { padding: 1.5rem; }

/* ---------- Stepper ---------- */
[class*="st-key-paso_card_"] { position: relative; }
[class*="st-key-ir_"] { position: absolute; inset: 0; z-index: 5; margin: 0; }
[class*="st-key-ir_"] button { width: 100%; height: 100%; opacity: 0; cursor: pointer; }
.cg-step { display: flex; align-items: center; gap: .6rem; padding: .6rem; border-radius: 10px;
  transition: background .15s; }
.cg-step:hover { background: var(--low); }
.cg-step.done { background: rgba(239,244,255,.6); }
.cg-step.active { background: rgba(220,233,255,.6); box-shadow: 0 0 0 2px rgba(0,81,213,.3); }
.cg-step.todo { opacity: .6; }
.cg-num { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; display: flex;
  align-items: center; justify-content: center; font-size: 13px; font-weight: 700; }
.done .cg-num { background: var(--ok); color: #fff; }
.active .cg-num { background: var(--secondary); color: #fff; }
.todo .cg-num { background: var(--container); color: var(--muted); }
.cg-steptxt { min-width: 0; }
.cg-steptop { display: flex; gap: 6px; align-items: baseline; white-space: nowrap; overflow: hidden; }
.cg-steplabel { font-size: 10.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.done .cg-steplabel { color: var(--ok); } .active .cg-steplabel { color: var(--secondary); }
.todo .cg-steplabel { color: var(--muted); }
.cg-stepname { font-size: 15px; font-weight: 600; color: var(--on-surface); overflow: hidden; text-overflow: ellipsis; }
.cg-stepsum { font-size: 12px; color: var(--muted); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; margin: 0; }

/* ---------- Formulario ---------- */
.cg-cardhead { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.cg-h2 { font-size: 20px; line-height: 28px; font-weight: 700; letter-spacing: -.015em;
  color: var(--on-surface); margin: 0; }
.cg-desc { font-size: 12.5px; color: var(--muted); margin: 2px 0 0; }
.cg-pill { font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  background: var(--container); color: var(--on-surface); padding: 4px 10px; border-radius: 999px;
  white-space: nowrap; }
.cg-hint { font-size: 11.5px; color: var(--muted); margin-top: -6px; }
.cg-badge { font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; }
.cg-rowlabel { display: flex; justify-content: space-between; align-items: center;
  font-size: 13px; font-weight: 600; color: var(--on-surface); gap: .5rem; flex-wrap: wrap; }
.cg-scale { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--muted);
  margin-top: -10px; }
.st-key-tenure_box { background: rgba(239,244,255,.6); border-radius: 12px; padding: 1rem; }
.st-key-paperless_box { background: rgba(239,244,255,.5); border-radius: 10px; padding: .75rem; }
.stApp label p { font-weight: 600; font-size: 13px; }

/* Tarjetas de selección (radio) */
.st-key-w_Contract [role="radiogroup"] {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .5rem; width: 100%;
}
.st-key-perfil [role="radiogroup"] { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; width: 100%; }
.st-key-w_Contract [role="radiogroup"] > div,
.st-key-perfil [role="radiogroup"] > div {
  background: var(--low); border-radius: 12px; padding: .7rem .9rem; margin: 0;
  cursor: pointer; transition: all .15s;
}
.st-key-w_Contract [role="radiogroup"] > div { padding: .9rem 1rem; }
.st-key-w_Contract [role="radiogroup"] > div:hover,
.st-key-perfil [role="radiogroup"] > div:hover { background: var(--container); }
.st-key-w_Contract [role="radiogroup"] > div[data-selected="true"],
.st-key-perfil [role="radiogroup"] > div[data-selected="true"] {
  background: rgba(220,233,255,.6); box-shadow: 0 0 0 2px var(--secondary);
}
.st-key-w_Contract [role="radiogroup"] label, .st-key-perfil [role="radiogroup"] label { width: 100%; margin: 0; }
.st-key-perfil [role="radiogroup"] label p { white-space: nowrap; }
.st-key-perfil .stRadio, .st-key-w_Contract.stRadio, .st-key-perfil [class*="st-key-w_"] { width: 100% !important; }
.st-key-w_Contract [role="radiogroup"] label p { font-size: 15px; font-weight: 600; }

/* Botones */
.stApp .stButton button { border-radius: 8px; border: none; background: var(--low);
  color: var(--on-surface); font-weight: 600; }
.stApp .stButton button:hover { background: var(--container); color: var(--on-surface); }
.st-key-evaluar button { background: #000 !important; color: #fff !important;
  box-shadow: 0 4px 10px rgba(0,0,0,.18); }
.st-key-evaluar button:hover { background: #213145 !important; }
.st-key-evaluar button [data-testid="stIconMaterial"] { color: #85f8c4; }
.st-key-otro button { background: var(--container); }

/* ---------- Panel de diagnóstico ---------- */
.st-key-panel { position: relative; overflow: hidden; }
.cg-glow { position: absolute; right: -48px; top: -48px; width: 160px; height: 160px;
  border-radius: 50%; filter: blur(32px); pointer-events: none; opacity: .35; }
.cg-panelhead { display: flex; justify-content: space-between; align-items: center; gap: .5rem;
  flex-wrap: wrap; position: relative; }
.cg-panelhead .t { display: flex; align-items: center; gap: 6px; font-size: 16px; font-weight: 700; }
.cg-risk { font-size: 10.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 4px; }
.cg-gaugebox { background: rgba(239,244,255,.6); border-radius: 12px; padding: 1rem .5rem .9rem;
  display: flex; flex-direction: column; align-items: center; margin-top: .75rem; }
.cg-gauge { position: relative; width: 220px; height: 124px; }
.cg-gauge img { width: 220px; height: 124px; display: block; }
.cg-gaugeval { position: absolute; bottom: 0; left: 0; right: 0; text-align: center; }
.cg-gaugeval .v { font-size: 34px; line-height: 38px; font-weight: 800; letter-spacing: -.03em; }
.cg-gaugeval .l { font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); }
.cg-class { font-size: 13px; font-weight: 700; margin-top: .6rem; }
.cg-delta { display: inline-flex; align-items: center; gap: 6px; background: #fff; border-radius: 999px;
  padding: 4px 12px; box-shadow: 0 1px 3px rgba(11,28,48,.1); margin-top: .5rem; font-size: 12px;
  color: var(--muted); }
.cg-delta b { font-size: 12px; }
.cg-action { background: rgba(220,233,255,.5); border-radius: 12px; padding: 1rem; margin-top: 1rem; }
.cg-action .h { display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 700;
  letter-spacing: .05em; text-transform: uppercase; color: var(--secondary); }
.cg-action p { font-size: 12.5px; line-height: 1.55; color: var(--on-surface); margin: 6px 0 0; }
.cg-factors { margin-top: 1rem; }
.cg-factors .h { display: flex; justify-content: space-between; font-size: 10.5px; font-weight: 700;
  letter-spacing: .05em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; gap: .5rem; }
.cg-factors .h span:last-child { color: var(--secondary); text-transform: none; letter-spacing: 0; font-weight: 500; }
.cg-factor { display: flex; justify-content: space-between; align-items: center; gap: .5rem;
  padding: 7px 9px; border-radius: 8px; background: var(--low); margin-bottom: 5px; font-size: 12.5px; }
.cg-factor:hover { background: var(--container); }
.cg-factor .n { display: flex; align-items: center; gap: 6px; min-width: 0; }
.cg-factor .s { font-weight: 700; font-size: 12px; white-space: nowrap; }
.cg-stale { font-size: 11.5px; color: #78350f; background: #fef3c7; border-radius: 8px;
  padding: 6px 10px; margin-top: .75rem; }
.cg-empty { text-align: center; padding: 2rem .5rem 1rem; color: var(--muted); }
.cg-empty .ic { width: 56px; height: 56px; border-radius: 50%; background: var(--low);
  color: var(--secondary); display: inline-flex; align-items: center; justify-content: center; }
.cg-empty b { display: block; color: var(--on-surface); font-size: 15px; margin: .75rem 0 .25rem; }
.cg-empty p { font-size: 12.5px; margin: 0; }

/* ---------- Pie de página ---------- */
.cg-footer { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  flex-wrap: wrap; border-top: 1px solid rgba(198,198,205,.4); padding-top: 1rem; margin-top: 1rem;
  font-size: 12px; color: var(--muted); }
.cg-footer .b { font-size: 11px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  color: var(--on-surface); margin-right: .5rem; }
.cg-footer .live { display: inline-flex; align-items: center; gap: 6px; }
.cg-footer .live i { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); display: inline-block; }

@media (max-width: 640px) {
  .st-key-w_Contract div[role="radiogroup"] { grid-template-columns: 1fr; }
  .cg-title { font-size: 23px; line-height: 30px; }
  .st-key-form_card, .st-key-panel { padding: 1rem; }
  .st-key-topbar { position: static; }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def icono(nombre: str, tam: int = 18, color: str | None = None) -> str:
    """Devuelve el HTML de un icono Material Symbols."""
    estilo = f"font-size:{tam}px;" + (f"color:{color};" if color else "")
    return f'<span class="msym" style="{estilo}">{nombre}</span>'


def texto(valor) -> str:
    """Traduce y escapa un valor del dataset para mostrarlo en HTML."""
    return html.escape(TRADUCCION.get(valor, str(valor)))


def valor_legible(feature: str, valor) -> str:
    """Formatea el valor de una feature para la interfaz."""
    if feature == "SeniorCitizen":
        return "Sí" if int(valor) == 1 else "No"
    if feature == "tenure":
        return f"{int(valor)} meses"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"${float(valor):,.2f}"
    return texto(valor)


def version_xgboost() -> str:
    """Versión instalada de xgboost (solo informativa)."""
    try:
        return version("xgboost")
    except PackageNotFoundError:
        return "—"


@st.cache_resource(show_spinner="Cargando modelo XGBoost...")
def obtener_modelo():
    """Carga el pipeline una sola vez por proceso (cacheado por Streamlit)."""
    return cargar_modelo(MODEL_PATH)


def calcular_factores(modelo, datos: dict, n: int = 5) -> list[dict]:
    """Contribuciones SHAP (TreeSHAP nativo de XGBoost) agregadas por variable.

    Devuelve las ``n`` variables con mayor impacto en la predicción de este
    cliente. Es puramente informativo: si falla, devuelve una lista vacía.
    """
    try:
        import xgboost as xgb

        preprocesador = modelo.named_steps["preprocessor"]
        booster = modelo.named_steps["model"].get_booster()
        matriz = preprocesador.transform(validar_entrada(datos))
        contribs = booster.predict(xgb.DMatrix(matriz), pred_contribs=True)[0][:-1]
        nombres = [n_.split("__", 1)[1] for n_ in preprocesador.get_feature_names_out()]

        por_variable: dict[str, float] = {}
        for nombre, valor in zip(nombres, contribs):
            if nombre in NUMERIC_FEATURES:
                feature = nombre
            else:
                feature = max((f for f in FEATURES if nombre.startswith(f + "_")), key=len)
            por_variable[feature] = por_variable.get(feature, 0.0) + float(valor)

        total = sum(abs(v) for v in por_variable.values()) or 1.0
        top = sorted(por_variable.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
        return [{"feature": f, "valor": v, "peso": abs(v) / total} for f, v in top]
    except Exception:  # noqa: BLE001 - la explicación nunca debe romper la app
        return []


def nivel_riesgo(prob: float, umbral: float) -> tuple[str, str, str]:
    """Devuelve (etiqueta del nivel, clave de color, recomendación)."""
    if prob >= 0.7:
        return ("Alto riesgo de abandono", "error",
                "<b>Riesgo alto.</b> Se recomienda contactar al cliente de inmediato con una "
                "oferta de retención (p. ej. migrar a contrato anual, descuentos o soporte técnico).")
    if prob >= umbral:
        return ("Riesgo moderado", "warning",
                "<b>Riesgo moderado.</b> El cliente muestra señales de abandono; conviene "
                "incluirlo en campañas preventivas de fidelización.")
    if prob >= 0.25:
        return ("Riesgo bajo-moderado", "info",
                "<b>Riesgo bajo-moderado.</b> Cliente relativamente estable, pero vale la pena "
                "monitorear cambios en su consumo o quejas.")
    return ("Bajo riesgo", "success",
            "<b>Riesgo bajo.</b> Cliente estable y leal; no requiere acciones de retención.")


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
def inicializar_estado() -> None:
    """Crea el estado inicial de la sesión."""
    st.session_state.setdefault("valores", dict(VALORES_POR_DEFECTO))
    st.session_state.setdefault("paso", 1)


def aplicar_valores(valores: dict) -> None:
    """Carga un conjunto completo de valores en el formulario."""
    st.session_state["valores"] = dict(valores)
    st.session_state.pop("resultado", None)


def al_elegir_caso() -> None:
    """Callback del selector de casos de ejemplo de la barra superior."""
    caso = st.session_state.get("caso")
    if caso == "Restablecer valores":
        aplicar_valores(VALORES_POR_DEFECTO)
    elif caso in EJEMPLOS:
        aplicar_valores(EJEMPLOS[caso])
    st.session_state["caso"] = None


def ir_a_paso(paso: int) -> None:
    """Cambia el paso activo del asistente."""
    st.session_state["paso"] = paso


def evaluar_otro() -> None:
    """Restablece el formulario para evaluar un nuevo cliente."""
    aplicar_valores(VALORES_POR_DEFECTO)
    ir_a_paso(1)


def sincronizar(feature: str) -> None:
    """Copia el valor de un widget al diccionario de valores del formulario."""
    valor = st.session_state[f"w_{feature}"]
    if feature == "PaperlessBilling":
        valor = "Yes" if valor else "No"
    st.session_state["valores"][feature] = valor


def preparar_widget(feature: str) -> str:
    """Inicializa el estado del widget con el valor actual y devuelve su clave."""
    clave = f"w_{feature}"
    valor = st.session_state["valores"][feature]
    st.session_state[clave] = (valor == "Yes") if feature == "PaperlessBilling" else valor
    return clave


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
def widget(feature: str, **kwargs) -> None:
    """Dibuja el control adecuado para una feature (estado sincronizado)."""
    clave = preparar_widget(feature)
    comun = {"key": clave, "on_change": sincronizar, "args": (feature,)}
    etiqueta = ETIQUETAS[feature]
    if feature == "SeniorCitizen":
        st.radio(etiqueta, [0, 1], horizontal=True,
                 format_func=lambda v: "Sí" if v == 1 else "No", **comun)
    elif feature == "tenure":
        st.slider(etiqueta, 0, 72, label_visibility="collapsed", **comun)
    elif feature in ("MonthlyCharges", "TotalCharges"):
        minimo, maximo = NUMERIC_RANGES[feature]
        st.number_input(f"{etiqueta} (USD)", min_value=float(minimo), max_value=float(maximo),
                        step=1.0, format="%.2f", **comun)
    elif feature == "PaperlessBilling":
        st.toggle("Facturación digital (Paperless)", **comun)
    elif feature == "Contract":
        opciones = CATEGORY_OPTIONS[feature]
        st.radio("Tipo de contrato comercial", opciones, horizontal=True,
                 format_func=lambda v: TRADUCCION.get(v, v),
                 captions=[f"{CAPTION_CONTRATO[o]} · {TASA_CONTRATO[o]} % churn histórico" for o in opciones],
                 **comun)
    elif kwargs.get("como_radio"):
        st.radio(etiqueta, CATEGORY_OPTIONS[feature], horizontal=True,
                 format_func=lambda v: TRADUCCION.get(v, v), **comun)
    else:
        st.selectbox(etiqueta, CATEGORY_OPTIONS[feature],
                     format_func=lambda v: TRADUCCION.get(v, v), **comun)


def badge_antiguedad(meses: int) -> str:
    """Etiqueta de color según la antigüedad del cliente."""
    if meses <= 6:
        fondo, color, txt = "#ffdad6", "#93000a", "Fase crítica de adopción"
    elif meses <= 24:
        fondo, color, txt = "#dce9ff", "#0b1c30", "Fidelización en curso"
    else:
        fondo, color, txt = "#85f8c4", "#002114", "Cliente consolidado"
    return (f'<span class="cg-badge" style="background:{fondo};color:{color}">'
            f"{meses} meses · {txt}</span>")


# --------------------------------------------------------------------------- #
# Secciones de la interfaz
# --------------------------------------------------------------------------- #
def barra_superior() -> None:
    """Marca, selector de casos de ejemplo e indicador del modelo."""
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([5, 3, 2.6], vertical_alignment="center")
        with c1:
            st.html(
                '<div class="cg-brandrow">'
                f'<div class="cg-logo">{icono("monitoring", 20)}</div>'
                '<span class="cg-brand">ChurnGuard Analytics</span>'
                '<span class="cg-tag">Predictor Telco</span></div>'
            )
        with c2:
            st.selectbox(
                "Cargar caso de ejemplo", [*EJEMPLOS, "Restablecer valores"],
                index=None, placeholder="⚡ Cargar caso de ejemplo", key="caso",
                on_change=al_elegir_caso, label_visibility="collapsed",
            )
        with c3:
            st.html('<div style="text-align:right"><span class="cg-live"><span class="cg-dot">'
                    '</span>XGBoost · ROC-AUC 0.849</span></div>')


def encabezado() -> float:
    """Título de la página y popover con el umbral. Devuelve el umbral."""
    c1, c2 = st.columns([4, 1.1], vertical_alignment="center")
    with c1:
        st.html(
            '<div class="cg-eyebrow"><span>Inferencia predictiva</span><span class="sep"></span>'
            '<span class="muted">Gradient Boosting · Telco Customer Churn (IBM)</span></div>'
            '<div class="cg-title">Evaluación de riesgo de abandono (Churn)</div>'
            '<p class="cg-sub">Flujo predictivo guiado con un modelo XGBoost entrenado con 7,043 '
            'clientes. Ingrese las variables del cliente o seleccione un caso de ejemplo.</p>'
        )
    with c2:
        umbral_actual = st.session_state.get("umbral", 0.50)
        with st.popover(f"Umbral {umbral_actual:.2f}", icon=":material/tune:",
                        width="stretch", key="params"):
            st.html(
                '<div class="cg-flyhead"><b>Configuración avanzada</b>'
                f'<span class="cg-pill">XGBoost {version_xgboost()}</span></div>'
            )
            umbral = st.slider("Umbral de decisión (threshold)", 0.05, 0.95, 0.50, 0.01,
                               key="umbral",
                               help="Probabilidad mínima para clasificar al cliente como Churn.")
            st.html('<div class="cg-scale" style="margin-top:-14px"><span>0.05 (sensible)</span>'
                    '<span>0.95 (conservador)</span></div>')
            filas = "".join(f"<div><span>{k}</span><b>{v}</b></div>"
                            for k, v in METRICAS_MODELO.items())
            st.html(f'<div class="cg-kv">{filas}</div>')
    return umbral


def resumen_paso(paso: int) -> str:
    """Resumen de una línea de los valores de cada paso."""
    v = st.session_state["valores"]
    if paso == 1:
        return (f"{texto(v['gender'])} · {'Senior' if int(v['SeniorCitizen']) else 'No senior'} · "
                f"{'Con pareja' if v['Partner'] == 'Yes' else 'Sin pareja'}")
    if paso == 2:
        return (f"{texto(v['Contract'])} · {int(v['tenure'])} meses · "
                f"${float(v['MonthlyCharges']):,.2f}/mes")
    return f"Internet: {texto(v['InternetService'])} · Teléfono: {texto(v['PhoneService'])}"


def stepper() -> None:
    """Indicador de progreso de 3 pasos (cada tarjeta es clicable)."""
    actual = st.session_state["paso"]
    with st.container(key="stepper"):
        columnas = st.columns(3, gap="small")
        for col, (paso, (nombre, _)) in zip(columnas, PASOS.items()):
            if paso < actual:
                estado, etiqueta, num = "done", f"Paso {paso}", icono("check", 18)
            elif paso == actual:
                estado, etiqueta, num = "active", "Paso activo", str(paso)
            else:
                estado, etiqueta, num = "todo", f"Paso {paso}", str(paso)
            with col, st.container(key=f"paso_card_{paso}"):
                st.html(
                    f'<div class="cg-step {estado}"><div class="cg-num">{num}</div>'
                    '<div class="cg-steptxt"><div class="cg-steptop">'
                    f'<span class="cg-steplabel">{etiqueta}</span>'
                    f'<span class="cg-stepname">{nombre}</span></div>'
                    f'<p class="cg-stepsum">{resumen_paso(paso)}</p></div></div>'
                )
                st.button(f"Ir al paso {paso}", key=f"ir_{paso}",
                          on_click=ir_a_paso, args=(paso,))


def cabecera_tarjeta(titulo: str, descripcion: str, paso: int) -> None:
    """Título de la tarjeta del formulario."""
    st.html(
        f'<div class="cg-cardhead"><div><h2 class="cg-h2">{titulo}</h2>'
        f'<p class="cg-desc">{descripcion}</p></div>'
        f'<span class="cg-pill">Paso {paso} / 3</span></div>'
    )


def paso_perfil() -> None:
    """Paso 1: datos demográficos."""
    cabecera_tarjeta("Perfil del cliente",
                     "Características demográficas y del hogar del titular de la cuenta.", 1)
    with st.container(key="perfil"):
        widget("gender", como_radio=True)
        widget("SeniorCitizen")
        widget("Partner", como_radio=True)
        widget("Dependents", como_radio=True)


def paso_cuenta() -> None:
    """Paso 2: cuenta, contrato y facturación."""
    v = st.session_state["valores"]
    cabecera_tarjeta("Variables de cuenta y facturación",
                     "Defina las condiciones comerciales para computar la probabilidad del cliente.", 2)
    with st.container(key="tenure_box"):
        st.html('<div class="cg-rowlabel"><span>Antigüedad del cliente (tenure)</span>'
                f'{badge_antiguedad(int(v["tenure"]))}</div>')
        widget("tenure")
        st.html('<div class="cg-scale"><span>0 meses (alta vulnerabilidad)</span>'
                '<span>24</span><span>48</span><span>72 meses (lealtad consolidada)</span></div>')

    widget("Contract")

    c1, c2 = st.columns(2)
    with c1:
        widget("PaymentMethod")
        st.html(f'<div class="cg-hint">Tasa histórica de abandono con este método: '
                f'<b>{TASA_PAGO[v["PaymentMethod"]]} %</b> (promedio general {TASA_CHURN_GLOBAL} %).</div>')
    with c2, st.container(key="paperless_box"):
        widget("PaperlessBilling")
        activa = v["PaperlessBilling"] == "Yes"
        st.html('<div class="cg-hint" style="margin-top:-8px">'
                f'{"Factura digital activada" if activa else "Factura en papel"} · '
                f'envío del resumen vía correo o portal.</div>')

    c3, c4 = st.columns(2)
    with c3:
        widget("MonthlyCharges")
        st.html(f'<div class="cg-hint">Media del dataset: ${MEDIA_CARGO_MENSUAL} / mes</div>')
    with c4:
        widget("TotalCharges")
        estimado = int(v["tenure"]) * float(v["MonthlyCharges"])
        st.html(f'<div class="cg-hint">Referencia: {int(v["tenure"])} meses × '
                f'${float(v["MonthlyCharges"]):,.2f} ≈ ${estimado:,.2f}</div>')


def paso_servicios() -> None:
    """Paso 3: servicios contratados."""
    cabecera_tarjeta("Servicios contratados",
                     "Conectividad y servicios de valor agregado activos en la cuenta.", 3)
    servicios = PASOS[3][1]
    columnas = st.columns(3)
    for i, feature in enumerate(servicios):
        with columnas[i % 3]:
            widget(feature)


def navegacion() -> bool:
    """Botones de navegación del asistente. Devuelve True si se pulsó evaluar."""
    paso = st.session_state["paso"]
    st.html('<div style="height:4px"></div>')
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="center")
    with c1:
        if paso > 1:
            st.button("Anterior", icon=":material/arrow_back:",
                      key="anterior", on_click=ir_a_paso, args=(paso - 1,), width="stretch")
    with c2:
        if paso < 3:
            st.button("Siguiente", icon=":material/arrow_forward:",
                      key="siguiente", on_click=ir_a_paso, args=(paso + 1,), width="stretch")
    with c3:
        return st.button("Evaluar riesgo de abandono", icon=":material/insights:",
                         key="evaluar", type="primary", width="stretch")


def gauge(prob: float, umbral: float, color: str) -> str:
    """SVG semicircular con la probabilidad y una marca en el umbral."""
    ang = math.pi * (1 - umbral)
    x1, y1 = 110 + 72 * math.cos(ang), 110 - 72 * math.sin(ang)
    x2, y2 = 110 + 104 * math.cos(ang), 110 - 104 * math.sin(ang)
    arco = "M 22 110 A 88 88 0 0 1 198 110"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 124">'
        f'<path d="{arco}" fill="none" stroke="#e5eeff" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="{arco}" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" '
        f'pathLength="100" stroke-dasharray="{max(prob * 100, 0.5):.2f} 100"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#0b1c30" '
        'stroke-width="2" stroke-dasharray="3 2"/></svg>'
    )
    datos = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{datos}" width="220" height="124" alt="">'


def panel_resultado(umbral: float) -> None:
    """Tarjeta lateral con el diagnóstico de la última evaluación."""
    resultado = st.session_state.get("resultado")
    with st.container(key="panel"):
        if resultado is None:
            st.html(
                '<div class="cg-panelhead"><span class="t">'
                f'{icono("query_stats", 20, "#0051d5")}Diagnóstico de retención</span></div>'
                f'<div class="cg-empty"><span class="ic">{icono("insights", 28)}</span>'
                '<b>Sin evaluación todavía</b>'
                '<p>Complete los pasos del formulario (o cargue un caso de ejemplo) y pulse '
                '<b style="display:inline;font-size:12.5px">Evaluar riesgo de abandono</b>.</p></div>'
            )
            return

        prob = resultado["probabilidad"]
        es_churn = prob >= umbral  # se recalcula por si el umbral cambió tras predecir
        nivel, clave_color, recomendacion = nivel_riesgo(prob, umbral)
        color, fondo, texto_color = COLORES[clave_color]
        delta = (prob - umbral) * 100
        clase = ("Churn · el cliente probablemente abandonará" if es_churn
                 else "No Churn · el cliente probablemente permanecerá")

        factores = ""
        for f in resultado.get("factores", []):
            sube = f["valor"] > 0
            c = "#ba1a1a" if sube else "#069669"
            valor = valor_legible(f["feature"], resultado["datos"][f["feature"]])
            factores += (
                '<div class="cg-factor"><span class="n">'
                f'{icono("arrow_upward" if sube else "arrow_downward", 16, c)}'
                f'<span>{ETIQUETAS[f["feature"]]}: {valor}</span></span>'
                f'<span class="s" style="color:{c}">{"↑" if sube else "↓"} '
                f'{f["peso"]:.0%}</span></div>'
            )
        bloque_factores = (
            '<div class="cg-factors"><div class="h"><span>Factores clave de esta predicción (SHAP)</span>'
            '<span>Peso relativo</span></div>' + factores + "</div>"
        ) if factores else ""

        aviso = ""
        if resultado.get("datos") != st.session_state["valores"]:
            aviso = ('<div class="cg-stale">Los datos del formulario cambiaron desde esta '
                     'evaluación. Pulse «Evaluar» de nuevo para actualizar.</div>')

        st.html(
            f'<div class="cg-glow" style="background:{fondo}"></div>'
            '<div class="cg-panelhead"><span class="t">'
            f'{icono("warning" if es_churn else "verified_user", 20, color)}Diagnóstico de retención</span>'
            f'<span class="cg-risk" style="background:{fondo};color:{texto_color}">{nivel}</span></div>'
            '<div class="cg-gaugebox"><div class="cg-gauge">'
            f'{gauge(prob, umbral, color)}'
            f'<div class="cg-gaugeval"><div class="v" style="color:{color}">{prob:.1%}</div>'
            '<div class="l">Probabilidad de churn</div></div></div>'
            f'<div class="cg-class" style="color:{texto_color}">{clase}</div>'
            f'<span class="cg-delta">{icono("north" if delta >= 0 else "south", 16, color)}'
            f'<b style="color:{color}">{delta:+.1f} pp</b>'
            f'{"sobre" if delta >= 0 else "bajo"} el umbral de decisión ({umbral:.2f})</span></div>'
            f'{aviso}'
            '<div class="cg-action"><div class="h">'
            f'{icono("verified", 16)}Acción recomendada</div><p>{recomendacion}</p></div>'
            f'{bloque_factores}'
        )
        c1, c2 = st.columns(2)
        with c1:
            st.button("Evaluar otro", icon=":material/refresh:", key="otro",
                      on_click=evaluar_otro, width="stretch")
        with c2:
            st.button("Modificar", icon=":material/edit:", key="modificar",
                      on_click=ir_a_paso, args=(1,), width="stretch")


def pie_pagina() -> None:
    """Pie de página informativo."""
    st.html(
        '<div class="cg-footer"><div><span class="b">ChurnGuard Analytics</span>'
        'Proyecto final · Inteligencia Artificial I</div>'
        '<div class="live"><i></i>Pipeline de inferencia activo · Telco Customer Churn (IBM)</div></div>'
    )


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
def main() -> None:
    """Punto de entrada de la aplicación."""
    st.html(CSS)
    inicializar_estado()
    barra_superior()
    umbral = encabezado()

    try:
        modelo = obtener_modelo()
    except FileNotFoundError:
        st.error("No se encontró `models/modelo.pkl`. Ejecuta primero "
                 "`notebooks/01_training.ipynb` para entrenar el modelo.")
        st.stop()
    except Exception as exc:  # pickle corrupto o versiones incompatibles
        st.error(f"No se pudo cargar el modelo: {exc}")
        st.stop()

    stepper()

    izquierda, derecha = st.columns([7, 5], gap="medium")
    with izquierda, st.container(key="form_card"):
        {1: paso_perfil, 2: paso_cuenta, 3: paso_servicios}[st.session_state["paso"]]()
        if navegacion():
            datos = dict(st.session_state["valores"])
            try:
                resultado = predecir(modelo, datos, umbral=umbral)
                resultado["datos"] = datos
                resultado["factores"] = calcular_factores(modelo, datos)
                st.session_state["resultado"] = resultado
            except ValueError as exc:
                st.session_state.pop("resultado", None)
                st.error(f"Datos de entrada inválidos: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("resultado", None)
                st.error(f"Ocurrió un error inesperado al predecir: {exc}")

    with derecha:
        panel_resultado(umbral)

    pie_pagina()


main()
