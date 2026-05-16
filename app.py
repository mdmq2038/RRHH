"""Aplicativo predictivo de Recursos Humanos para escritorio y celular.

Ejecuta con:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="RRHH Predictivo",
    page_icon="RR",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "edad",
    "antiguedad_anios",
    "horas_capacitacion",
    "ausencias_12m",
    "satisfaccion",
    "evaluacion_desempeno",
    "horas_extra_mes",
    "distancia_km",
    "salario_mensual",
    "proyectos_activos",
]

MODEL_LABELS = {
    "attrition": "Riesgo de rotacion",
    "performance": "Desempeno esperado",
    "promotion": "Probabilidad de promocion",
    "salary": "Salario recomendado",
}


@dataclass(frozen=True)
class PredictionResult:
    title: str
    value: str
    detail: str
    level: str


@st.cache_data(show_spinner=False)
def build_hr_dataset(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Crea una base de ejemplo realista para entrenar modelos demostrativos."""
    rng = np.random.default_rng(seed)
    edad = rng.integers(20, 61, rows)
    antiguedad = np.clip(rng.gamma(2.1, 2.2, rows), 0, 20).round(1)
    horas_capacitacion = np.clip(rng.normal(34, 18, rows), 0, 120).round(1)
    ausencias = rng.poisson(4.2, rows)
    satisfaccion = np.clip(rng.normal(3.4, 0.9, rows), 1, 5).round(1)
    evaluacion = np.clip(rng.normal(3.5, 0.75, rows), 1, 5).round(1)
    horas_extra = np.clip(rng.normal(12, 8, rows), 0, 55).round(1)
    distancia = np.clip(rng.gamma(2.0, 6.5, rows), 1, 60).round(1)
    proyectos = np.clip(rng.poisson(3.2, rows), 1, 9)
    salario = (
        1400
        + edad * 22
        + antiguedad * 110
        + evaluacion * 280
        + proyectos * 90
        + rng.normal(0, 260, rows)
    ).round(0)

    rotacion_score = (
        -1.45
        - satisfaccion * 0.72
        - antiguedad * 0.11
        + ausencias * 0.12
        + horas_extra * 0.045
        + distancia * 0.025
        - evaluacion * 0.16
        + (salario < np.percentile(salario, 35)) * 0.75
    )
    rotacion_prob = 1 / (1 + np.exp(-rotacion_score))
    rota = rng.binomial(1, np.clip(rotacion_prob, 0.03, 0.92))

    desempeno = np.clip(
        45
        + evaluacion * 9.5
        + satisfaccion * 4.2
        + horas_capacitacion * 0.11
        - ausencias * 1.25
        - horas_extra * 0.12
        + proyectos * 1.8
        + rng.normal(0, 5, rows),
        0,
        100,
    ).round(1)

    promotion_score = (
        -5.4
        + evaluacion * 0.78
        + satisfaccion * 0.26
        + antiguedad * 0.13
        + horas_capacitacion * 0.018
        + proyectos * 0.19
        - ausencias * 0.09
    )
    promotion_prob = 1 / (1 + np.exp(-promotion_score))
    promocion = rng.binomial(1, np.clip(promotion_prob, 0.02, 0.85))

    return pd.DataFrame(
        {
            "edad": edad,
            "antiguedad_anios": antiguedad,
            "horas_capacitacion": horas_capacitacion,
            "ausencias_12m": ausencias,
            "satisfaccion": satisfaccion,
            "evaluacion_desempeno": evaluacion,
            "horas_extra_mes": horas_extra,
            "distancia_km": distancia,
            "salario_mensual": salario,
            "proyectos_activos": proyectos,
            "rota": rota,
            "desempeno_100": desempeno,
            "promocion": promocion,
        }
    )


@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Entrena modelos predictivos para las decisiones principales de RRHH."""
    x = data[FEATURES]
    models: dict[str, Pipeline] = {}
    metrics = []

    tasks = {
        "attrition": (
            data["rota"],
            RandomForestClassifier(n_estimators=180, random_state=7, max_depth=8),
            "clasificacion",
        ),
        "promotion": (
            data["promocion"],
            RandomForestClassifier(n_estimators=180, random_state=9, max_depth=8),
            "clasificacion",
        ),
        "performance": (
            data["desempeno_100"],
            RandomForestRegressor(n_estimators=180, random_state=11, max_depth=9),
            "regresion",
        ),
        "salary": (
            data["salario_mensual"],
            RandomForestRegressor(n_estimators=180, random_state=13, max_depth=9),
            "regresion",
        ),
    }

    for key, (target, estimator, task_type) in tasks.items():
        x_train, x_test, y_train, y_test = train_test_split(
            x, target, test_size=0.22, random_state=21
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        models[key] = pipeline

        if task_type == "clasificacion":
            metric_name = "Exactitud"
            metric_value = accuracy_score(y_test, predicted)
        else:
            metric_name = "R2"
            metric_value = r2_score(y_test, predicted)

        metrics.append(
            {
                "Modelo": MODEL_LABELS[key],
                "Metrica": metric_name,
                "Valor": round(float(metric_value), 3),
                "MAE": round(float(mean_absolute_error(y_test, predicted)), 2)
                if task_type == "regresion"
                else "-",
            }
        )

    return models, pd.DataFrame(metrics)


def apply_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }
        .hero {
            background: linear-gradient(135deg, #17324d 0%, #2563eb 55%, #38bdf8 100%);
            color: white;
            padding: 28px;
            border-radius: 24px;
            margin-bottom: 18px;
        }
        .hero h1 {font-size: 2.25rem; margin-bottom: .35rem;}
        .hero p {font-size: 1.05rem; opacity: .95; margin-bottom: 0;}
        .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 12px;
        }
        @media (max-width: 768px) {
            .main .block-container {padding-left: .8rem; padding-right: .8rem;}
            .hero {padding: 20px; border-radius: 18px;}
            .hero h1 {font-size: 1.55rem;}
            .hero p {font-size: .95rem;}
            div[data-testid="stMetric"] {padding: 12px;}
            [data-testid="stSidebar"] {min-width: 80vw;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def collect_employee_inputs() -> pd.DataFrame:
    st.sidebar.header("Datos del colaborador")
    st.sidebar.caption("Ajusta los datos y observa las predicciones en tiempo real.")

    edad = st.sidebar.slider("Edad", 18, 65, 32)
    antiguedad = st.sidebar.slider("Antiguedad en anios", 0.0, 25.0, 3.5, 0.5)
    horas_capacitacion = st.sidebar.slider("Horas de capacitacion anual", 0, 140, 38)
    ausencias = st.sidebar.slider("Ausencias en 12 meses", 0, 35, 4)
    satisfaccion = st.sidebar.slider("Satisfaccion laboral (1-5)", 1.0, 5.0, 3.6, 0.1)
    evaluacion = st.sidebar.slider("Evaluacion de desempeno (1-5)", 1.0, 5.0, 3.8, 0.1)
    horas_extra = st.sidebar.slider("Horas extra al mes", 0, 70, 10)
    distancia = st.sidebar.slider("Distancia al trabajo (km)", 1, 80, 12)
    salario = st.sidebar.slider("Salario mensual", 800, 9000, 3200, 50)
    proyectos = st.sidebar.slider("Proyectos activos", 1, 10, 3)

    return pd.DataFrame(
        [
            {
                "edad": edad,
                "antiguedad_anios": antiguedad,
                "horas_capacitacion": horas_capacitacion,
                "ausencias_12m": ausencias,
                "satisfaccion": satisfaccion,
                "evaluacion_desempeno": evaluacion,
                "horas_extra_mes": horas_extra,
                "distancia_km": distancia,
                "salario_mensual": salario,
                "proyectos_activos": proyectos,
            }
        ]
    )


def classify_probability(probability: float, reverse: bool = False) -> str:
    if reverse:
        if probability >= 0.66:
            return "Alto"
        if probability >= 0.38:
            return "Medio"
        return "Bajo"

    if probability >= 0.66:
        return "Critico"
    if probability >= 0.38:
        return "Moderado"
    return "Bajo"


def build_predictions(models: dict[str, Pipeline], employee: pd.DataFrame) -> list[PredictionResult]:
    attrition_prob = float(models["attrition"].predict_proba(employee)[0][1])
    promotion_prob = float(models["promotion"].predict_proba(employee)[0][1])
    performance = float(models["performance"].predict(employee)[0])
    salary = float(models["salary"].predict(employee)[0])

    return [
        PredictionResult(
            "Riesgo de rotacion",
            f"{attrition_prob:.0%}",
            "Prioriza acciones de retencion si el nivel es moderado o critico.",
            classify_probability(attrition_prob),
        ),
        PredictionResult(
            "Desempeno esperado",
            f"{performance:.1f}/100",
            "Estimacion basada en capacitacion, evaluacion, ausencias y carga laboral.",
            "Alto" if performance >= 78 else "Medio" if performance >= 62 else "Bajo",
        ),
        PredictionResult(
            "Probabilidad de promocion",
            f"{promotion_prob:.0%}",
            "Identifica talento listo para planes de carrera y sucesion.",
            classify_probability(promotion_prob, reverse=True),
        ),
        PredictionResult(
            "Salario recomendado",
            f"${salary:,.0f}",
            "Referencia interna para equidad salarial y bandas compensatorias.",
            "Referencia",
        ),
    ]


def retention_recommendations(employee: pd.DataFrame, attrition_level: str) -> list[str]:
    row = employee.iloc[0]
    recommendations = []

    if attrition_level in {"Moderado", "Critico"}:
        recommendations.append("Agendar una conversacion de permanencia durante los proximos 7 dias.")
    if row["satisfaccion"] < 3.3:
        recommendations.append("Disenar un plan de mejora de clima, reconocimiento y motivadores individuales.")
    if row["horas_extra_mes"] > 18:
        recommendations.append("Revisar carga laboral, turnos y balance vida-trabajo para reducir fatiga.")
    if row["ausencias_12m"] > 8:
        recommendations.append("Activar acompanamiento de bienestar y revisar posibles causas de ausentismo.")
    if row["horas_capacitacion"] < 24:
        recommendations.append("Asignar capacitacion tecnica o de liderazgo alineada al rol.")
    if not recommendations:
        recommendations.append("Mantener seguimiento trimestral y reforzar reconocimiento por desempeno.")

    return recommendations


def render_prediction_cards(predictions: list[PredictionResult]) -> None:
    columns = st.columns(4)
    for column, prediction in zip(columns, predictions):
        column.metric(prediction.title, prediction.value, prediction.level)
        column.caption(prediction.detail)


def render_dashboard(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    employee: pd.DataFrame,
    predictions: list[PredictionResult],
) -> None:
    tab_predict, tab_analytics, tab_methods = st.tabs(
        ["Prediccion individual", "Analitica RRHH", "Metodos predictivos"]
    )

    with tab_predict:
        render_prediction_cards(predictions)
        st.subheader("Plan de accion recomendado")
        for item in retention_recommendations(employee, predictions[0].level):
            st.success(item)

        st.subheader("Comparativo del colaborador vs. poblacion")
        compare = pd.DataFrame(
            {
                "Indicador": ["Satisfaccion", "Desempeno", "Ausencias", "Horas extra", "Capacitacion"],
                "Colaborador": [
                    employee["satisfaccion"].iloc[0],
                    employee["evaluacion_desempeno"].iloc[0],
                    employee["ausencias_12m"].iloc[0],
                    employee["horas_extra_mes"].iloc[0],
                    employee["horas_capacitacion"].iloc[0],
                ],
                "Promedio empresa": [
                    data["satisfaccion"].mean(),
                    data["evaluacion_desempeno"].mean(),
                    data["ausencias_12m"].mean(),
                    data["horas_extra_mes"].mean(),
                    data["horas_capacitacion"].mean(),
                ],
            }
        )
        st.plotly_chart(
            px.bar(compare, x="Indicador", y=["Colaborador", "Promedio empresa"], barmode="group"),
            use_container_width=True,
        )

    with tab_analytics:
        left, right = st.columns([1, 1])
        left.subheader("Distribucion de desempeno")
        left.plotly_chart(px.histogram(data, x="desempeno_100", nbins=24, color="rota"), use_container_width=True)
        right.subheader("Factores asociados a rotacion")
        right.plotly_chart(
            px.scatter(
                data,
                x="satisfaccion",
                y="horas_extra_mes",
                color="rota",
                size="ausencias_12m",
                hover_data=["edad", "antiguedad_anios", "salario_mensual"],
            ),
            use_container_width=True,
        )
        st.subheader("Metricas de validacion")
        st.dataframe(metrics, hide_index=True, use_container_width=True)

    with tab_methods:
        st.markdown(
            """
            <div class="card">
            <h3>Modelos incluidos</h3>
            <ul>
              <li><b>Clasificacion:</b> estima rotacion y promocion con Random Forest.</li>
              <li><b>Regresion:</b> proyecta desempeno y salario recomendado.</li>
              <li><b>Validacion:</b> divide la base en entrenamiento y prueba para mostrar metricas.</li>
              <li><b>Uso responsable:</b> los resultados apoyan decisiones humanas; no reemplazan entrevistas, politicas ni revision etica.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(data.head(50), hide_index=True, use_container_width=True)


def main() -> None:
    apply_responsive_styles()
    data = build_hr_dataset()
    models, metrics = train_models(data)

    st.markdown(
        """
        <div class="hero">
          <h1>RRHH Predictivo en Python</h1>
          <p>Aplicativo responsive para PC y celular que predice rotacion, desempeno, promocion y salario recomendado.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    employee = collect_employee_inputs()
    predictions = build_predictions(models, employee)
    render_dashboard(data, metrics, employee, predictions)

    st.caption(
        "Nota: esta demostracion usa datos sinteticos. Para produccion, conecta datos reales, anonimiza informacion sensible y audita sesgos."
    )


if __name__ == "__main__":
    main()
