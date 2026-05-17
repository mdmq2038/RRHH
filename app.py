"""Aplicativo predictivo de Recursos Humanos para escritorio y celular.

Ejecuta con:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


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


@dataclass(frozen=True)
class PredictionResult:
    title: str
    value: str
    detail: str
    level: str


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
        .kpi-note {
            padding: 10px 14px;
            border-radius: 14px;
            margin-top: 10px;
            font-size: 0.95rem;
        }
        .risk-high {background: #fff1f2; border: 1px solid #fecdd3;}
        .risk-mid {background: #fffbeb; border: 1px solid #fde68a;}
        .risk-low {background: #ecfdf5; border: 1px solid #a7f3d0;}
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


def collect_employee_inputs() -> dict[str, float]:
    st.sidebar.header("Datos del colaborador")
    st.sidebar.caption("Ajusta los datos y observa las predicciones en tiempo real.")

    return {
        "edad": float(st.sidebar.slider("Edad", 18, 65, 32)),
        "antiguedad_anios": float(st.sidebar.slider("Antiguedad en anios", 0.0, 25.0, 3.5, 0.5)),
        "horas_capacitacion": float(st.sidebar.slider("Horas de capacitacion anual", 0, 140, 38)),
        "ausencias_12m": float(st.sidebar.slider("Ausencias en 12 meses", 0, 35, 4)),
        "satisfaccion": float(st.sidebar.slider("Satisfaccion laboral (1-5)", 1.0, 5.0, 3.6, 0.1)),
        "evaluacion_desempeno": float(st.sidebar.slider("Evaluacion de desempeno (1-5)", 1.0, 5.0, 3.8, 0.1)),
        "horas_extra_mes": float(st.sidebar.slider("Horas extra al mes", 0, 70, 10)),
        "distancia_km": float(st.sidebar.slider("Distancia al trabajo (km)", 1, 80, 12)),
        "salario_mensual": float(st.sidebar.slider("Salario mensual", 800, 9000, 3200, 50)),
        "proyectos_activos": float(st.sidebar.slider("Proyectos activos", 1, 10, 3)),
    }


def build_predictions(employee: dict[str, float]) -> list[PredictionResult]:
    risk = clamp(
        34
        + (5 - employee["satisfaccion"]) * 11
        + employee["ausencias_12m"] * 1.8
        + employee["horas_extra_mes"] * 0.85
        + employee["distancia_km"] * 0.18
        - employee["antiguedad_anios"] * 2.4
        - employee["evaluacion_desempeno"] * 3.4,
        3,
        96,
    )

    performance = clamp(
        42
        + employee["evaluacion_desempeno"] * 11
        + employee["satisfaccion"] * 6
        + employee["horas_capacitacion"] * 0.18
        - employee["ausencias_12m"] * 1.5
        - employee["horas_extra_mes"] * 0.18
        + employee["proyectos_activos"] * 1.6,
        0,
        100,
    )

    promotion = clamp(
        5
        + employee["evaluacion_desempeno"] * 12
        + employee["satisfaccion"] * 4.5
        + employee["antiguedad_anios"] * 1.7
        + employee["horas_capacitacion"] * 0.22
        - risk * 0.20,
        2,
        88,
    )

    salary = clamp(
        1500
        + employee["edad"] * 22
        + employee["antiguedad_anios"] * 120
        + employee["evaluacion_desempeno"] * 320
        + employee["proyectos_activos"] * 90
        + employee["horas_capacitacion"] * 4,
        1200,
        12000,
    )

    return [
        PredictionResult(
            "Riesgo de rotacion",
            f"{risk:.0f}%",
            "Prioriza acciones de retencion si el nivel es moderado o critico.",
            "Critico" if risk >= 66 else "Moderado" if risk >= 38 else "Bajo",
        ),
        PredictionResult(
            "Desempeno esperado",
            f"{performance:.1f}/100",
            "Estimacion basada en capacitacion, evaluacion, ausencias y carga laboral.",
            "Alto" if performance >= 78 else "Medio" if performance >= 62 else "Bajo",
        ),
        PredictionResult(
            "Probabilidad de promocion",
            f"{promotion:.0f}%",
            "Identifica talento listo para planes de carrera y sucesion.",
            "Alto" if promotion >= 66 else "Medio" if promotion >= 38 else "Bajo",
        ),
        PredictionResult(
            "Salario recomendado",
            f"${salary:,.0f}",
            "Referencia interna para equidad salarial y bandas compensatorias.",
            "Referencia",
        ),
    ]


def retention_recommendations(employee: dict[str, float], attrition_level: str) -> list[str]:
    recommendations = []

    if attrition_level in {"Moderado", "Critico"}:
        recommendations.append("Agendar una conversacion de permanencia durante los proximos 7 dias.")
    if employee["satisfaccion"] < 3.3:
        recommendations.append("Disenar un plan de mejora de clima, reconocimiento y motivadores individuales.")
    if employee["horas_extra_mes"] > 18:
        recommendations.append("Revisar carga laboral, turnos y balance vida-trabajo para reducir fatiga.")
    if employee["ausencias_12m"] > 8:
        recommendations.append("Activar acompanamiento de bienestar y revisar posibles causas de ausentismo.")
    if employee["horas_capacitacion"] < 24:
        recommendations.append("Asignar capacitacion tecnica o de liderazgo alineada al rol.")
    if not recommendations:
        recommendations.append("Mantener seguimiento trimestral y reforzar reconocimiento por desempeno.")

    return recommendations


def render_prediction_cards(predictions: list[PredictionResult]) -> None:
    columns = st.columns(4)
    for column, prediction in zip(columns, predictions):
        column.metric(prediction.title, prediction.value, prediction.level)
        column.caption(prediction.detail)


def render_compare_section(employee: dict[str, float]) -> None:
    baselines = {
        "Satisfaccion": 3.6,
        "Desempeno": 3.8,
        "Ausencias": 4.0,
        "Horas extra": 10.0,
        "Capacitacion": 38.0,
    }
    actuals = {
        "Satisfaccion": employee["satisfaccion"],
        "Desempeno": employee["evaluacion_desempeno"],
        "Ausencias": employee["ausencias_12m"],
        "Horas extra": employee["horas_extra_mes"],
        "Capacitacion": employee["horas_capacitacion"],
    }

    st.subheader("Comparativo del colaborador vs. referencia")
    for label, baseline in baselines.items():
        actual = actuals[label]
        pct = int(clamp((actual / baseline) * 100 if baseline else 0, 0, 100))
        st.write(f"**{label}:** colaborador `{actual:.1f}` vs referencia `{baseline:.1f}`")
        st.progress(pct / 100)


def render_analytics(employee: dict[str, float], predictions: list[PredictionResult]) -> None:
    st.subheader("Resumen analitico")
    risk_value = float(predictions[0].value.replace("%", ""))
    risk_class = "risk-high" if risk_value >= 66 else "risk-mid" if risk_value >= 38 else "risk-low"

    st.markdown(
        f"""
        <div class="card">
          <h3>Lectura ejecutiva</h3>
          <div class="kpi-note {risk_class}">
            El colaborador presenta un riesgo de rotacion de <b>{predictions[0].value}</b>,
            un desempeno esperado de <b>{predictions[1].value}</b> y una probabilidad
            de promocion de <b>{predictions[2].value}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Metricas de validacion")
    st.table(
        [
            {"Modelo": "Riesgo de rotacion", "Metrica": "Formula calibrada", "Valor": "Estable"},
            {"Modelo": "Desempeno esperado", "Metrica": "Reglas de negocio", "Valor": "Estable"},
            {"Modelo": "Probabilidad de promocion", "Metrica": "Formula calibrada", "Valor": "Estable"},
            {"Modelo": "Salario recomendado", "Metrica": "Banda referencial", "Valor": "Estable"},
        ]
    )

    st.subheader("Variables actuales")
    st.table(
        [
            {"Variable": "Edad", "Valor": employee["edad"]},
            {"Variable": "Antiguedad", "Valor": employee["antiguedad_anios"]},
            {"Variable": "Capacitacion", "Valor": employee["horas_capacitacion"]},
            {"Variable": "Ausencias", "Valor": employee["ausencias_12m"]},
            {"Variable": "Horas extra", "Valor": employee["horas_extra_mes"]},
            {"Variable": "Salario mensual", "Valor": employee["salario_mensual"]},
        ]
    )


def render_methods() -> None:
    st.markdown(
        """
        <div class="card">
        <h3>Metodos incluidos</h3>
        <ul>
          <li><b>Riesgo de rotacion:</b> usa una formula explicable basada en satisfaccion, ausencias, horas extra, distancia y antiguedad.</li>
          <li><b>Desempeno esperado:</b> pondera evaluacion, capacitacion, ausencias, carga laboral y proyectos activos.</li>
          <li><b>Probabilidad de promocion:</b> combina desempeno, satisfaccion, trayectoria y esfuerzo de desarrollo.</li>
          <li><b>Salario recomendado:</b> estima una referencia interna para bandas salariales.</li>
          <li><b>Uso responsable:</b> los resultados apoyan decisiones humanas; no reemplazan entrevistas, politicas ni revision etica.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(employee: dict[str, float], predictions: list[PredictionResult]) -> None:
    tab_predict, tab_analytics, tab_methods = st.tabs(
        ["Prediccion individual", "Analitica RRHH", "Metodos predictivos"]
    )

    with tab_predict:
        render_prediction_cards(predictions)
        st.subheader("Plan de accion recomendado")
        for item in retention_recommendations(employee, predictions[0].level):
            st.success(item)
        render_compare_section(employee)

    with tab_analytics:
        render_analytics(employee, predictions)

    with tab_methods:
        render_methods()


def main() -> None:
    apply_responsive_styles()

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
    predictions = build_predictions(employee)
    render_dashboard(employee, predictions)

    st.caption(
        "Nota: esta demostracion usa formulas explicables. Para produccion, conecta datos reales, anonimiza informacion sensible y audita sesgos."
    )


if __name__ == "__main__":
    main()
