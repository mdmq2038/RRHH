from pathlib import Path

import pytest


REQUIRED_FEATURES = [
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


def test_app_source_declares_required_hr_features():
    source = Path("app.py").read_text(encoding="utf-8")

    for feature in REQUIRED_FEATURES:
        assert feature in source
    assert "RRHH Predictivo en Python" in source
    assert "@media (max-width: 768px)" in source


def test_requirements_include_app_dependencies():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    for dependency in ["streamlit", "pandas", "numpy", "scikit-learn", "plotly"]:
        assert dependency in requirements


def test_models_return_predictions_when_dependencies_are_available():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    pytest.importorskip("sklearn")

    import app

    data = app.build_hr_dataset(rows=180, seed=321)
    models, metrics = app.train_models(data)
    employee = pd.DataFrame(
        [
            {
                "edad": 35,
                "antiguedad_anios": 4,
                "horas_capacitacion": 40,
                "ausencias_12m": 3,
                "satisfaccion": 4.0,
                "evaluacion_desempeno": 4.2,
                "horas_extra_mes": 8,
                "distancia_km": 10,
                "salario_mensual": 3500,
                "proyectos_activos": 3,
            }
        ]
    )

    predictions = app.build_predictions(models, employee)

    assert set(models) == {"attrition", "promotion", "performance", "salary"}
    assert len(metrics) == 4
    assert len(predictions) == 4
    assert all(prediction.value for prediction in predictions)
